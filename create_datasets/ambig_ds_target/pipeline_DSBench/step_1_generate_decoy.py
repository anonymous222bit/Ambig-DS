#!/usr/bin/env python3
"""Calibrated strong target-ambiguity decoy generator (v2).

Derived from step_3b_strong_decoy_generator.py with two changes intended to
address findings from the V2 (inferability) and V3 (noise sensitivity)
validation audits:

  Fix A — PER-TASK NOISE CALIBRATION
      Instead of a fixed noise/swap rate (10% in the original), bisect the
      noise level per task so that |cv_decoy - cv_true| <= --cv_tolerance.
      This collapses the per-task cv_ratio distribution around 1.0, which
      directly defeats the inverted-CV attack (Selectors B/C in the
      inferability audit). Falls back to the original fixed rate if
      bisection fails to find a level meeting the tolerance.

  Fix B — DTYPE / ROUNDING MATCH
      After noise is applied, snap the decoy column back to the same dtype
      and decimal precision as the truth column. Prevents an LLM (or human)
      from spotting the decoy because, e.g., the truth is integer 0/1 and
      the noised decoy is a continuous float. Marginal-match-exact is
      preserved (or recomputed) after this step.

All outputs land at --out_root and never overwrite the original
final_data_v3/target_ambig/data/ tree unless the user explicitly points
--out_root there.
"""
from __future__ import annotations

import os

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


ID_LIKE = {
    "id", "Id", "ID", "row_id", "ForecastId", "PassengerId",
    "textID", "qa_id", "essay_id", "datetime", "date",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def detect_id_col(cols):
    for c in cols:
        if c in ID_LIKE:
            return c
    return None


def encode_train_for_model(train_df, feat_cols, dtype="float32"):
    """Numeric encoding of the train feature matrix only (test features are
    written verbatim to the output and are never needed for diagnostics or
    decoy construction). Object columns -> integer codes; NaNs preserved.
    Uses float32 to halve memory on large tasks.
    """
    cols = []
    for c in feat_cols:
        s = train_df[c]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            cols.append(pd.Categorical(s.astype("string")).codes.astype(dtype))
        else:
            cols.append(pd.to_numeric(s, errors="coerce").astype(dtype).values)
    return np.column_stack(cols)


def quick_cv(X, y, target_type, n_folds=3, max_rows=50_000, seed=0):
    """Returns CV score on a subsample.
       - binary classification     -> ROC-AUC
       - multiclass classification -> accuracy
       - regression                -> R^2
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n > max_rows:
        idx = rng.choice(n, max_rows, replace=False)
        X = X[idx]
        y = np.asarray(y)[idx]
    else:
        y = np.asarray(y)

    if target_type == "classification":
        # encode labels
        y_enc = pd.Series(y).astype("category").cat.codes.values
        n_classes = int(pd.Series(y).nunique())
        is_binary = n_classes == 2
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, te in skf.split(X, y_enc):
            m = HistGradientBoostingClassifier(
                max_iter=80, max_depth=5, random_state=seed,
            )
            m.fit(X[tr], y_enc[tr])
            if is_binary:
                p = m.predict_proba(X[te])[:, 1]
                try:
                    s = roc_auc_score(y_enc[te], p)
                except ValueError:
                    s = float("nan")
            else:
                yhat = m.predict(X[te])
                s = accuracy_score(y_enc[te], yhat)
            scores.append(s)
        return float(np.nanmean(scores))

    # regression
    y = np.asarray(y, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        m = HistGradientBoostingRegressor(
            max_iter=80, max_depth=5, random_state=seed,
        )
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.nanmean(scores))


def per_feature_abs_spearman(X, y, target_type, max_rows=50_000, seed=0):
    """Per-feature |Spearman| with y. Subsamples for speed."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    if n > max_rows:
        idx = rng.choice(n, max_rows, replace=False)
        Xs = X[idx]
        ys = np.asarray(y)[idx]
    else:
        Xs = X
        ys = np.asarray(y)
    if target_type == "classification":
        ys = pd.Series(ys).astype("category").cat.codes.values
    out = np.zeros(p)
    for j in range(p):
        col = Xs[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 10:
            out[j] = 0.0
            continue
        try:
            r = stats.spearmanr(col[mask], ys[mask]).correlation
            out[j] = 0.0 if r is None or np.isnan(r) else abs(r)
        except Exception:
            out[j] = 0.0
    return out


def build_decoy(X_num, y, target_type, seed,
                low_corr_pool_size=None,
                low_corr_pool_frac=0.5,
                pool_min=3,
                pool_max=10):
    """Construct a decoy column with the same marginal distribution as `y`,
    that is feature-predictable but mostly orthogonal to `y`.

    Method: rank-map the marginal of `y` onto a rank score derived from a
    pool of features that have LOW |Spearman| correlation with `y`. Picking
    low-corr features ensures `corr(val_2, y)` is small, while the
    rank-mapping itself ensures `val_2` is highly predictable from those
    same features.
    """
    rng = np.random.default_rng(seed)
    n, p = X_num.shape

    abs_corrs = per_feature_abs_spearman(
        X_num, y, target_type=target_type, seed=seed,
    )
    sorted_idx = np.argsort(abs_corrs)  # ascending
    if low_corr_pool_size is None:
        k = max(pool_min, min(pool_max, int(p * low_corr_pool_frac)))
    else:
        k = max(pool_min, min(pool_max, int(low_corr_pool_size), p))
    pool = sorted_idx[:k].tolist()

    # standardized feature score from the low-corr pool
    Xp = pd.DataFrame(X_num[:, pool])
    Xp = Xp.fillna(Xp.median())
    Xp = (Xp - Xp.mean()) / (Xp.std(ddof=0) + 1e-9)
    score = Xp.sum(axis=1).values
    # tiny jitter to break ties deterministically per seed
    score = score + rng.normal(0, 1e-9, n)

    # rank-map y's marginal onto the score order
    order = np.argsort(score)
    sorted_y = np.sort(np.asarray(y))
    val_2 = np.empty_like(sorted_y)
    val_2[order] = sorted_y
    return val_2, pool, abs_corrs[pool].tolist()


def add_label_noise_classification(val_2, frac, seed):
    """Swap `frac` of labels via random pairing. Preserves marginal exactly."""
    if frac <= 0:
        return val_2.copy()
    rng = np.random.default_rng(seed)
    n = len(val_2)
    n_swaps = int(n * frac / 2)
    if n_swaps <= 0:
        return val_2.copy()
    out = val_2.copy()
    perm = rng.permutation(n)[: 2 * n_swaps]
    for i in range(0, 2 * n_swaps, 2):
        a, b = perm[i], perm[i + 1]
        out[a], out[b] = out[b], out[a]
    return out


def add_label_noise_regression(val_2, sigma_frac, seed):
    """Add gaussian jitter to the rank score, then re-rank-map onto the
    sorted target distribution. Preserves marginal exactly.
    """
    if sigma_frac <= 0:
        return val_2.copy()
    rng = np.random.default_rng(seed)
    n = len(val_2)
    sd = float(np.std(val_2)) + 1e-12
    jittered = val_2 + rng.normal(0, sd * sigma_frac, n)
    order = np.argsort(jittered)
    sorted_vals = np.sort(val_2)
    out = np.empty_like(val_2)
    out[order] = sorted_vals
    return out


# ---------------------------------------------------------------------------
# Fix A: per-task noise calibration (binary search on noise level)
# ---------------------------------------------------------------------------

def calibrate_noise(val_2_raw, X_num, y_true, target_type, seed,
                    cv_true, cv_tolerance, max_cv_rows,
                    n_steps=8, lo=0.0, hi=0.5):
    """Bisect noise level so that |cv(decoy) - cv_true| <= cv_tolerance.

    Returns (val_2_calibrated, chosen_level, cv_decoy_at_chosen,
             trace) where trace is a list of (level, cv_decoy) pairs from
    the bisection, useful for diagnostics.

    Strategy
    --------
    The pre-noise decoy (val_2_raw) is by construction MORE learnable than
    the truth (rank-mapped from a feature-derived score). Adding noise
    monotonically reduces cv_decoy. We therefore binary-search the smallest
    level whose cv_decoy <= cv_true + cv_tolerance, while also bounding
    cv_decoy >= cv_true - cv_tolerance.

    If cv_true is invalid (NaN or 0) we fall back to the original behaviour
    by returning val_2_raw with level=0.0 and the caller's responsibility
    to apply a default noise rate.
    """
    if cv_true is None or np.isnan(cv_true) or cv_true == 0:
        return None, None, None, []

    def make(level):
        if target_type == "classification":
            return add_label_noise_classification(val_2_raw, level, seed)
        return add_label_noise_regression(val_2_raw, level, seed)

    def cv_at(level):
        v = make(level)
        return v, quick_cv(X_num, v, target_type=target_type,
                           max_rows=max_cv_rows, seed=seed)

    trace = []
    # First check the boundaries.
    v_lo, cv_lo = cv_at(lo)
    trace.append((lo, cv_lo))
    if abs(cv_lo - cv_true) <= cv_tolerance:
        return v_lo, lo, cv_lo, trace
    v_hi, cv_hi = cv_at(hi)
    trace.append((hi, cv_hi))
    # If even the highest noise can't bring cv_decoy down to cv_true, the
    # decoy is hopelessly more learnable than the truth: take the highest
    # noise and let the caller report the residual gap.
    if cv_hi > cv_true + cv_tolerance:
        return v_hi, hi, cv_hi, trace
    # Symmetrically: if even zero noise undershoots cv_true (decoy less
    # learnable than truth), bisection won't help; return the closer of
    # the two endpoints.
    if cv_lo < cv_true - cv_tolerance:
        if abs(cv_hi - cv_true) < abs(cv_lo - cv_true):
            return v_hi, hi, cv_hi, trace
        return v_lo, lo, cv_lo, trace

    # Bisect: invariant cv(lo) >= cv_true + tol >= cv(hi).
    best_v, best_level, best_cv = v_hi, hi, cv_hi
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        v_mid, cv_mid = cv_at(mid)
        trace.append((mid, cv_mid))
        if abs(cv_mid - cv_true) < abs(best_cv - cv_true):
            best_v, best_level, best_cv = v_mid, mid, cv_mid
        if cv_mid > cv_true + cv_tolerance:
            lo = mid  # still too learnable -> need more noise
        elif cv_mid < cv_true - cv_tolerance:
            hi = mid  # too noisy -> back off
        else:
            return v_mid, mid, cv_mid, trace
    return best_v, best_level, best_cv, trace


# ---------------------------------------------------------------------------
# Fix B: snap decoy back to truth's dtype / decimal precision
# ---------------------------------------------------------------------------

def _infer_decimal_precision(y, max_check=50_000, cap=6):
    """Return a small integer indicating how many decimal places the truth
    column appears to use (e.g. 0 for integers, 2 for currency-like).

    Cheap heuristic: take up to `max_check` finite values, find the smallest
    nonzero |x - round(x, k)| over k = 0..cap, and report the k with the
    largest fraction of exactly-representable values.
    """
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    if a.size > max_check:
        a = a[:max_check]
    for k in range(0, cap + 1):
        rounded = np.round(a, k)
        # 1e-9 tolerates float64 round-trip noise
        if np.all(np.abs(a - rounded) < 1e-9):
            return k
    return cap


def snap_to_truth_dtype(val_2, y_true, target_type):
    """Make the decoy column visually indistinguishable from the truth at
    the dtype / precision / value-set level. Marginal-preservation is
    re-checked after this snap and if it's broken, returns the un-snapped
    decoy.
    """
    y_arr = np.asarray(y_true)
    v_arr = np.asarray(val_2)

    if target_type == "classification":
        # add_label_noise_classification permutes existing labels, so dtype
        # is already preserved. Defensive: if dtypes drifted, cast to truth.
        try:
            return v_arr.astype(y_arr.dtype, copy=False)
        except Exception:
            return v_arr

    # Regression: match (a) integer-ness, (b) decimal precision.
    y_f = np.asarray(y_true, dtype=float)
    if pd.api.types.is_integer_dtype(pd.Series(y_true)) or np.all(
        np.isfinite(y_f) & (y_f == np.round(y_f))
    ):
        snapped = np.rint(v_arr.astype(float)).astype(y_arr.dtype, copy=False)
    else:
        k = _infer_decimal_precision(y_f)
        if k is None:
            return v_arr
        snapped = np.round(v_arr.astype(float), k)

    # Marginal-preservation can break under rounding. We require the
    # decoy multiset to equal the truth multiset EXACTLY (not just
    # np.allclose, which has rtol=1e-5 and silently passes large shifts
    # on data with big magnitudes). If the snap broke the marginal, fall
    # back to the un-snapped decoy.
    if not np.array_equal(np.sort(np.asarray(snapped, dtype=float)),
                          np.sort(y_f)):
        return v_arr
    return snapped


def diag_correlation(y_true, y_decoy, target_type):
    if target_type == "classification":
        # accuracy as a "how predictive of true is the decoy" sanity check
        return ("match_rate", float((np.asarray(y_true) == np.asarray(y_decoy)).mean()))
    return ("pearson", float(np.corrcoef(np.asarray(y_true, dtype=float),
                                         np.asarray(y_decoy, dtype=float))[0, 1]))


def diag_marginal(y_true, y_decoy, target_type):
    if target_type == "classification":
        a = pd.Series(y_true).value_counts().sort_index()
        b = pd.Series(y_decoy).value_counts().sort_index()
        return bool(a.equals(b))
    # Regression: require EXACT multiset equality on the float values.
    # np.allclose has rtol=1e-5 which silently passes large shifts on
    # large-magnitude targets; we want strict equality so that any
    # divergence from "marginals match exactly by construction" is
    # flagged for the downstream filter.
    return bool(np.array_equal(np.sort(np.asarray(y_true, dtype=float)),
                               np.sort(np.asarray(y_decoy, dtype=float))))


def write_sample_submission(test_df, idcol, target_type, y_train_for_default, out_path):
    if target_type == "classification":
        default = pd.Series(y_train_for_default).mode().iloc[0]
    else:
        default = float(np.mean(y_train_for_default))
    pd.DataFrame({idcol: test_df[idcol].values, "prediction": default}).to_csv(
        out_path, index=False
    )


# ---------------------------------------------------------------------------
# per-task pipeline
# ---------------------------------------------------------------------------

def process_task(task_row, args, master_rng):
    task = task_row["task"]
    target = task_row["target_name"]
    target_type = task_row["target_type"].strip().lower()
    if target_type not in ("classification", "regression"):
        raise ValueError(f"unsupported target_type={target_type} for task={task}")

    src = Path(args.src_data_root) / task
    # If a train-row cap is requested, use nrows= so we never load the
    # full file into memory (avoids OOM on cgroup-capped sessions).
    # max_train_rows: 0 (or None) disables the cap; positive int caps train
    # to first N rows. Defaults to 500_000 to match the HF release.
    cap = args.max_train_rows if args.max_train_rows else None
    train_kwargs = {"low_memory": False}
    if cap is not None:
        train_kwargs["nrows"] = cap
    train = pd.read_csv(src / "train.csv", **train_kwargs)
    test = pd.read_csv(src / "test.csv", low_memory=False)
    if cap is not None and len(train) == cap:
        print(f"  [subsample] capped train at first {len(train):,} rows")

    # If the source is one of the previous v3 packages, the original target
    # column was renamed (to val_1, col_A, val_7, ...) and a weak decoy column
    # was appended. Read the source manifest if present to recover the
    # rename mapping; otherwise fall back to a small set of common names.
    if target not in train.columns:
        renamed_to = None
        manifest_p = src / "_manifest.json"
        if manifest_p.exists():
            try:
                m = json.loads(manifest_p.read_text())
                for op in m.get("data_operations", []):
                    if op.get("type") == "rename_columns" and op.get("path") == "train.csv":
                        mapping = op.get("mapping", {})
                        if target in mapping and mapping[target] in train.columns:
                            renamed_to = mapping[target]
                            break
            except Exception:
                renamed_to = None
        if renamed_to is None:
            for cand in ("val_1", "col_A", "val_7"):
                if cand in train.columns:
                    renamed_to = cand
                    break
        if renamed_to is not None:
            train = train.rename(columns={renamed_to: target})
            # drop any weak decoy(s): all train-only columns that aren't id/feat/target
            test_cols_local = set(pd.read_csv(src / "test.csv", nrows=0).columns)
            extras = [c for c in train.columns
                      if c != target and c not in test_cols_local and c not in ID_LIKE]
            if extras:
                train = train.drop(columns=extras)

    if target not in train.columns:
        raise RuntimeError(f"target {target!r} not found in {src/'train.csv'}")

    idcol = detect_id_col(test.columns)
    if idcol is None:
        # fabricate a stable id if none present
        idcol = "id"
        train = train.copy()
        test = test.copy()
        train[idcol] = np.arange(len(train))
        test[idcol] = np.arange(len(test))

    feat_cols = [c for c in train.columns
                 if c != idcol and c != target and c in test.columns]
    if not feat_cols:
        raise RuntimeError(f"{task}: no shared feature columns between train and test")

    # ---- anonymize feature names ----------------------------------------
    width = max(2, len(str(len(feat_cols))))
    feat_map = {orig: f"f_{i + 1:0{width}d}" for i, orig in enumerate(feat_cols)}
    anon_feat_cols = [feat_map[c] for c in feat_cols]

    # ---- encode train only (test features are passed through verbatim) --
    X_tr_num = encode_train_for_model(train, feat_cols)
    y_tr = train[target].values

    # ---- diagnostic CV on the true target -------------------------------
    cv_true = quick_cv(X_tr_num, y_tr, target_type=target_type,
                       max_rows=args.max_cv_rows, seed=args.seed)

    # ---- build decoy ----------------------------------------------------
    decoy_seed = int(master_rng.integers(0, 2**31 - 1))

    # Special case: if the TRUE target is essentially unlearnable from the
    # visible features (cv_true near zero), the rank-mapping construction
    # would produce a decoy that IS learnable from those same features
    # (cv_decoy ~ 0.5+) because we project the truth marginal onto a
    # feature-derived score. The result is a CV-gap > 0.5 that no level of
    # noise can close without destroying the marginal. In that regime the
    # paper's "indistinguishable by learnability" claim demands cv_decoy
    # also near zero, which we achieve by taking a random permutation of
    # the truth values: this preserves the marginal exactly, has zero
    # systematic correlation with features in expectation, and matches
    # cv_true ~ 0 by construction.
    use_random_permutation = bool(
        cv_true is not None and not np.isnan(cv_true)
        and cv_true < args.unlearnable_truth_threshold
    )
    if use_random_permutation:
        print(f"  [unlearnable-truth] {task}: cv_true={cv_true:.3f} < "
              f"{args.unlearnable_truth_threshold}; using random "
              f"permutation of truth as decoy.")
        rng_perm = np.random.default_rng(decoy_seed)
        val_2_raw = np.asarray(y_tr).copy()
        rng_perm.shuffle(val_2_raw)
        # Empty pool / zero correlations (no feature-based construction).
        pool_idx = np.array([], dtype=int)
        pool_corrs = []
    else:
        val_2_raw, pool_idx, pool_corrs = build_decoy(
            X_tr_num, y_tr, target_type=target_type, seed=decoy_seed,
            pool_min=args.pool_min, pool_max=args.pool_max,
            low_corr_pool_frac=args.pool_frac,
        )
    # ---- Fix A: per-task noise calibration ------------------------------
    # Bisect noise level so |cv(decoy) - cv_true| <= cv_tolerance.
    fallback_noise = (args.noise_classification if target_type == "classification"
                      else args.noise_regression)
    calibrated_v2, calibrated_level, calibrated_cv, cal_trace = calibrate_noise(
        val_2_raw, X_tr_num, y_tr, target_type=target_type,
        seed=decoy_seed, cv_true=cv_true,
        cv_tolerance=args.cv_tolerance, max_cv_rows=args.max_cv_rows,
        n_steps=args.bisection_steps, lo=0.0, hi=args.max_noise,
    )

    if calibrated_v2 is None:
        # cv_true was unusable; fall back to the original fixed-noise path.
        if target_type == "classification":
            val_2 = add_label_noise_classification(val_2_raw, fallback_noise, decoy_seed)
        else:
            val_2 = add_label_noise_regression(val_2_raw, fallback_noise, decoy_seed)
        chosen_level = fallback_noise
        bisection_converged = False
    else:
        val_2 = calibrated_v2
        chosen_level = calibrated_level
        bisection_converged = (
            cv_true is not None and not np.isnan(cv_true)
            and abs(calibrated_cv - cv_true) <= args.cv_tolerance
        )

    # If the decoy ended up less learnable than the truth even at the
    # calibrated level, expand the feature pool (preserve original safety
    # net). Threshold 0.75: paper Sec 4.2 requires |cv_decoy - cv_true|
    # small in BOTH directions; if cv_decoy < 0.75*cv_true, the gap is
    # already > 0.25*cv_true and the bisection can't close it without a
    # more learnable raw decoy.
    cv_decoy = quick_cv(X_tr_num, val_2, target_type=target_type,
                        max_rows=args.max_cv_rows, seed=args.seed)
    if not use_random_permutation and cv_true and cv_decoy < 0.75 * cv_true:
        print(f"  [retry] {task}: weak decoy (cv_ratio={cv_decoy / cv_true:.2f}), "
              f"expanding pool to ALL features")
        decoy_seed += 1
        val_2_raw, pool_idx, pool_corrs = build_decoy(
            X_tr_num, y_tr, target_type=target_type, seed=decoy_seed,
            pool_min=X_tr_num.shape[1], pool_max=X_tr_num.shape[1],
            low_corr_pool_frac=1.0,
        )
        calibrated_v2, calibrated_level, calibrated_cv, cal_trace = calibrate_noise(
            val_2_raw, X_tr_num, y_tr, target_type=target_type,
            seed=decoy_seed, cv_true=cv_true,
            cv_tolerance=args.cv_tolerance, max_cv_rows=args.max_cv_rows,
            n_steps=args.bisection_steps, lo=0.0, hi=args.max_noise,
        )
        if calibrated_v2 is None:
            if target_type == "classification":
                val_2 = add_label_noise_classification(val_2_raw, fallback_noise, decoy_seed)
            else:
                val_2 = add_label_noise_regression(val_2_raw, fallback_noise, decoy_seed)
            chosen_level = fallback_noise
            bisection_converged = False
        else:
            val_2 = calibrated_v2
            chosen_level = calibrated_level
            bisection_converged = (
                cv_true is not None and not np.isnan(cv_true)
                and abs(calibrated_cv - cv_true) <= args.cv_tolerance
            )
        cv_decoy = quick_cv(X_tr_num, val_2, target_type=target_type,
                            max_rows=args.max_cv_rows, seed=args.seed)

    # ---- Fix B: snap decoy back to truth dtype / precision --------------
    if args.apply_dtype_snap:
        val_2 = snap_to_truth_dtype(val_2, y_tr, target_type)
        # Recompute cv after the snap (rounding can shift it slightly).
        cv_decoy = quick_cv(X_tr_num, val_2, target_type=target_type,
                            max_rows=args.max_cv_rows, seed=args.seed)

    corr_metric, corr_value = diag_correlation(y_tr, val_2, target_type)
    marginal_ok = diag_marginal(y_tr, val_2, target_type)

    # ---- explicit correlation filter (paper §4.2) ----------------------
    # Paper text: "the remaining tasks retain the closest candidate satisfying
    # the marginal-match and low-correlation filters." Marginal is exact by
    # construction; this block enforces the correlation half. If the current
    # decoy exceeds --max_abs_correlation, we resample alternative decoy seeds
    # and prefer the candidate that (i) is within the cap and (ii) has the
    # smallest |cv_decoy - cv_true|. If no candidate satisfies the cap, we
    # keep the candidate with the smallest |corr|.
    if abs(corr_value) > args.max_abs_correlation:
        print(f"  [corr-filter] |{corr_metric}|={abs(corr_value):.3f} > "
              f"{args.max_abs_correlation}; resampling decoy seed "
              f"({args.corr_filter_seeds} attempts)")
        # tuple: (val_2, seed, level, cv_decoy, corr, pool_idx, pool_corrs)
        best = (val_2, decoy_seed, chosen_level, cv_decoy, corr_value,
                pool_idx, pool_corrs)
        for k in range(args.corr_filter_seeds):
            cand_seed = int(master_rng.integers(0, 2**31 - 1))
            v_raw, p_idx, p_corrs = build_decoy(
                X_tr_num, y_tr, target_type=target_type, seed=cand_seed,
                pool_min=args.pool_min, pool_max=args.pool_max,
                low_corr_pool_frac=args.pool_frac,
            )
            v2_cal, lvl_cal, cv_cal, _ = calibrate_noise(
                v_raw, X_tr_num, y_tr, target_type=target_type,
                seed=cand_seed, cv_true=cv_true,
                cv_tolerance=args.cv_tolerance, max_cv_rows=args.max_cv_rows,
                n_steps=args.bisection_steps, lo=0.0, hi=args.max_noise,
            )
            if v2_cal is None:
                v2_cal = (
                    add_label_noise_classification(v_raw, fallback_noise, cand_seed)
                    if target_type == "classification"
                    else add_label_noise_regression(v_raw, fallback_noise, cand_seed)
                )
                lvl_cal = fallback_noise
                cv_cal = quick_cv(X_tr_num, v2_cal, target_type=target_type,
                                  max_rows=args.max_cv_rows, seed=args.seed)
            if args.apply_dtype_snap:
                v2_cal = snap_to_truth_dtype(v2_cal, y_tr, target_type)
            _, c_cal = diag_correlation(y_tr, v2_cal, target_type)
            cur_in = abs(best[4]) <= args.max_abs_correlation
            new_in = abs(c_cal) <= args.max_abs_correlation
            if new_in and (not cur_in
                           or abs(cv_cal - cv_true) < abs(best[3] - cv_true)):
                best = (v2_cal, cand_seed, lvl_cal, cv_cal, c_cal,
                        p_idx, p_corrs)
            elif (not cur_in) and abs(c_cal) < abs(best[4]):
                best = (v2_cal, cand_seed, lvl_cal, cv_cal, c_cal,
                        p_idx, p_corrs)
        (val_2, decoy_seed, chosen_level, cv_decoy, corr_value,
         pool_idx, pool_corrs) = best
        marginal_ok = diag_marginal(y_tr, val_2, target_type)
        bisection_converged = (
            cv_true is not None and not np.isnan(cv_true)
            and abs(cv_decoy - cv_true) <= args.cv_tolerance
        )
    correlation_filter_passed = bool(
        abs(corr_value) <= args.max_abs_correlation
    )

    # ---- final marginal-preservation guard ------------------------------
    # All upstream paths (build_decoy, add_label_noise_*, snap_to_truth_dtype)
    # are written to preserve the truth marginal exactly. This is a defensive
    # post-condition: if any future change to those paths regresses, fall
    # back to the un-snapped, un-corr-filtered noised decoy (which is
    # marginal-correct by construction) so the paper's Sec 4.2 "marginals
    # match by construction" claim cannot be silently violated.
    if not diag_marginal(y_tr, val_2, target_type):
        print(f"  [marginal-guard] {task}: post-pipeline marginal broken; "
              f"regenerating from val_2_raw + fallback noise.")
        if target_type == "classification":
            val_2 = add_label_noise_classification(
                val_2_raw, fallback_noise, decoy_seed)
        else:
            val_2 = add_label_noise_regression(
                val_2_raw, fallback_noise, decoy_seed)
        # snap with strict guard returns un-snapped on failure
        if args.apply_dtype_snap:
            val_2 = snap_to_truth_dtype(val_2, y_tr, target_type)
        cv_decoy = quick_cv(X_tr_num, val_2, target_type=target_type,
                            max_rows=args.max_cv_rows, seed=args.seed)
        corr_metric, corr_value = diag_correlation(y_tr, val_2, target_type)
        marginal_ok = diag_marginal(y_tr, val_2, target_type)
        correlation_filter_passed = bool(
            abs(corr_value) <= args.max_abs_correlation)

    # ---- random truth ordering ------------------------------------------
    # Use a slug-derived seed so the val_1/val_2 assignment is independent
    # of the calibration/decoy RNG state. Otherwise, when many tasks share
    # the same `--seed` (the default in build/rebuild), `master_rng` reaches
    # the same state by the time of this draw and every task ends up with
    # truth=val_1, which lets a trivial "always val_1" or "lower-entropy
    # tie-break" heuristic identify the truth column (paper Sec 4.2 audit).
    import hashlib
    slug_seed = int(
        hashlib.blake2b(task.encode("utf-8"), digest_size=8).hexdigest(), 16
    ) ^ int(args.seed)
    truth_first = bool(np.random.default_rng(slug_seed).integers(0, 2))
    truth_col = "val_1" if truth_first else "val_2"
    decoy_col = "val_2" if truth_first else "val_1"

    # ---- assemble output frames -----------------------------------------
    out_train = pd.DataFrame({idcol: train[idcol].values})
    for orig, anon in zip(feat_cols, anon_feat_cols):
        out_train[anon] = train[orig].values
    out_train[truth_col] = y_tr
    out_train[decoy_col] = val_2
    # canonical column order: id, features..., val_1, val_2
    out_train = out_train[[idcol] + anon_feat_cols + ["val_1", "val_2"]]

    out_test = pd.DataFrame({idcol: test[idcol].values})
    for orig, anon in zip(feat_cols, anon_feat_cols):
        out_test[anon] = test[orig].values
    out_test = out_test[[idcol] + anon_feat_cols]

    # ---- write everything -----------------------------------------------
    task_dir = Path(args.out_root) / "data" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    out_train.to_csv(task_dir / "train.csv", index=False)
    out_test.to_csv(task_dir / "test.csv", index=False)
    write_sample_submission(
        test, idcol, target_type, y_tr,
        task_dir / "sample_submission.csv",
    )

    manifest = {
        "task": task,
        "true_target_column": truth_col,
        "decoy_column": decoy_col,
        "original_target_name": target,
        "target_type": target_type,
        "id_column": idcol,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(feat_cols),
        "feature_map": feat_map,
        "anon_feature_columns": anon_feat_cols,
        "decoy_method": (
            "random_permutation_unlearnable_truth"
            if use_random_permutation
            else (
                "rank_map_lowcorr_pool+calibrated_noise+dtype_match"
                if args.apply_dtype_snap
                else "rank_map_lowcorr_pool+calibrated_noise"
            )
        ),
        "decoy_pool_anon_features": [anon_feat_cols[i] for i in pool_idx],
        "decoy_pool_abs_spearman_with_truth": pool_corrs,
        "noise_classification_frac": args.noise_classification,
        "noise_regression_sigma_frac": args.noise_regression,
        "calibration": {
            "calibrated_noise_level": (
                float(chosen_level) if chosen_level is not None else None
            ),
            "bisection_converged": bool(bisection_converged),
            "cv_tolerance": args.cv_tolerance,
            "bisection_steps": args.bisection_steps,
            "max_noise": args.max_noise,
            "trace": [
                {"level": float(lv), "cv_decoy": float(cv) if cv is not None else None}
                for lv, cv in cal_trace
            ],
        },
        "diagnostics": {
            "cv_true": cv_true,
            "cv_decoy": cv_decoy,
            "cv_ratio_decoy_over_true": (
                cv_decoy / cv_true if cv_true and not np.isnan(cv_true) else None
            ),
            "correlation_metric": corr_metric,
            "correlation_truth_vs_decoy": corr_value,
            "marginal_match_exact": marginal_ok,
            "correlation_filter_passed": correlation_filter_passed,
            "max_abs_correlation": args.max_abs_correlation,
        },
        "seeds": {"master": args.seed, "decoy": decoy_seed},
    }
    (task_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "task": task,
        "n_train": len(train),
        "n_features": len(feat_cols),
        "target_type": target_type,
        "truth_col": truth_col,
        "cv_true": cv_true,
        "cv_decoy": cv_decoy,
        "cv_ratio": (cv_decoy / cv_true) if cv_true else float("nan"),
        "corr_metric": corr_metric,
        "corr_value": corr_value,
        "marginal_ok": marginal_ok,
        "decoy_pool_size": len(pool_idx),
        "calibrated_noise_level": (
            float(chosen_level) if chosen_level is not None else float("nan")
        ),
        "bisection_converged": bool(bisection_converged),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks_csv", required=True,
                   help="Pilot subset CSV (e.g. final_data_v3/target_ambig/"
                        "target_ambiguity_tasks.csv).")
    p.add_argument("--src_data_root",
                   default=os.path.join(
                       os.environ.get("AMBIG_DSBENCH_ROOT", "."),
                       "Dataset/data_modeling/data/data/data_resplit"),
                   help="Root containing per-task train.csv/test.csv.")
    p.add_argument("--out_root", required=True,
                   help="Output root, e.g. final_data_v3/target_ambig.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_classification", type=float, default=0.10,
                   help="Fallback fraction of decoy labels to swap when "
                        "calibration cannot run (e.g. cv_true is NaN).")
    p.add_argument("--noise_regression", type=float, default=0.10,
                   help="Fallback regression-noise sigma when calibration "
                        "cannot run.")
    p.add_argument("--cv_tolerance", type=float, default=0.02,
                   help="Fix A: target |cv_decoy - cv_true| <= this value "
                        "during per-task noise calibration.")
    p.add_argument("--bisection_steps", type=int, default=8,
                   help="Fix A: max bisection iterations for noise calibration.")
    p.add_argument("--max_noise", type=float, default=0.8,
                   help="Fix A: upper bound on the bisected noise level. "
                        "Raised from 0.5 to 0.8 so that calibration can "
                        "reach decoy CV close to truth CV on tasks with a "
                        "strong feature-target signal (paper Sec 4.2 "
                        "requires |cv_decoy - cv_true| small).")
    p.add_argument("--unlearnable_truth_threshold", type=float, default=0.10,
                   help="If quick_cv on the true target is below this "
                        "value, the truth is essentially unlearnable from "
                        "the visible features. In that regime the rank-"
                        "mapped decoy would be MORE learnable than the "
                        "truth (cv_decoy >> 0), violating the Sec 4.2 "
                        "indistinguishability claim. We instead use a "
                        "random permutation of truth values as the decoy: "
                        "marginal is preserved exactly, cv_decoy ~ 0 by "
                        "construction, and the gap closes.")
    p.add_argument("--apply_dtype_snap", action="store_true",
                   help="Fix B (off by default): snap decoy column to truth's "
                        "dtype/decimal precision after noise. The 10-task "
                        "trial showed this does not move LLM-attack accuracy "
                        "because the LLM reads dataset semantics, not dtype.")
    p.add_argument("--max_abs_correlation", type=float, default=0.30,
                   help="Paper §4.2 correlation filter: reject decoys with "
                        "|corr(y, decoy)| above this cap and resample seeds. "
                        "Set high (e.g. 1.0) to disable.")
    p.add_argument("--corr_filter_seeds", type=int, default=8,
                   help="Max alternative decoy seeds to try when the "
                        "correlation filter rejects the initial candidate.")
    p.add_argument("--pool_frac", type=float, default=0.7,
                   help="Fraction of features (sorted by ascending |corr| "
                        "with the truth) to include in the decoy pool.")
    p.add_argument("--pool_min", type=int, default=4)
    p.add_argument("--pool_max", type=int, default=40)
    p.add_argument("--max_cv_rows", type=int, default=50_000,
                   help="Subsample cap for CV diagnostics (speed).")
    p.add_argument("--max_train_rows", type=int, default=500_000,
                   help="Cap the training set to first N rows. Default 500_000 "
                        "matches the published HF release (Ambig-DS-T), which "
                        "caps a handful of multi-million-row tasks (e.g. "
                        "ventilator-pressure-prediction, tps-dec-2021) so the "
                        "release stays redistribution-friendly and CV "
                        "diagnostics stay tractable. Pass --max_train_rows 0 "
                        "to disable the cap. Test set is never capped.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N tasks of the pilot CSV.")
    p.add_argument("--only", default=None,
                   help="Comma-separated task names to process (overrides "
                        "--limit; useful for re-running individual tasks).")
    p.add_argument("--force", action="store_true",
                   help="Regenerate tasks even when an existing _manifest.json "
                        "is found in the output dir. Default: skip already-done tasks.")
    args = p.parse_args()

    pilot = pd.read_csv(args.tasks_csv)
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        pilot = pilot[pilot["task"].isin(wanted)].reset_index(drop=True)
        if pilot.empty:
            sys.exit("--only matched no tasks in the pilot CSV")
    if args.limit:
        pilot = pilot.head(args.limit).reset_index(drop=True)

    out_root = Path(args.out_root)
    (out_root / "data").mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    rows = []
    t_start = time.time()
    for i, row in pilot.iterrows():
        task = row["task"]
        elapsed = time.time() - t_start
        print(f"\n[{i + 1}/{len(pilot)}] {task}  (elapsed {elapsed:.0f}s)", flush=True)
        out_task_dir = out_root / "data" / task
        manifest_out = out_task_dir / "_manifest.json"
        if not args.force and manifest_out.exists():
            try:
                m = json.loads(manifest_out.read_text())
                d = m.get("diagnostics", {})
                rows.append({
                    "task": task,
                    "truth_col": m.get("true_target_column", ""),
                    "cv_true": d.get("cv_true", float('nan')),
                    "cv_decoy": d.get("cv_decoy", float('nan')),
                    "cv_ratio": d.get("cv_ratio_decoy_over_true", float('nan')),
                    "corr_metric": d.get("correlation_metric", ""),
                    "corr_value": d.get("correlation_truth_vs_decoy", float('nan')),
                    "marginal_ok": d.get("marginal_match_exact", ""),
                    "decoy_pool_size": len(m.get("decoy_pool_anon_features", [])),
                    "n_train": m.get("n_train", ""),
                    "n_features": m.get("n_features", ""),
                    "target_type": m.get("target_type", ""),
                })
                print(f"  [skip] already generated -> {manifest_out}", flush=True)
                continue
            except Exception as e:
                print(f"  [skip-failed] could not read existing manifest: {e}; regenerating", flush=True)
        try:
            r = process_task(row, args, master_rng)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append({"task": task, "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(r)
        print(
            f"  truth={r['truth_col']:>5}  "
            f"cv_true={r['cv_true']:.3f}  cv_decoy={r['cv_decoy']:.3f}  "
            f"ratio={r['cv_ratio']:.2f}  "
            f"{r['corr_metric']}={r['corr_value']:+.3f}  "
            f"marginal_ok={r['marginal_ok']}  "
            f"pool={r['decoy_pool_size']}",
            flush=True,
        )

    diag = pd.DataFrame(rows)
    diag_path = out_root / "_decoy_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    print(f"\nWrote diagnostics: {diag_path}")
    if "cv_ratio" in diag.columns and not diag["cv_ratio"].isna().all():
        print("\n=== ambiguity quality (target: cv_ratio in [0.85, 1.10] AND |corr| low) ===")
        print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
