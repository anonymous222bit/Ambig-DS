#!/usr/bin/env python3
"""Step V1 — decoy-quality validation for target-ambiguity tasks.

For every task in the benchmark dir produced by step_1_setup_benchmark.py:
    <benchmark-dir>/data/<slug>/ambig/{train.csv, _manifest.json}
loads `train.csv` and `_manifest.json`, recovers the (true_target, decoy)
pair from the manifest, and computes:

  Distributional similarity
    * marginal_match_exact      — multiset equality of y vs y_decoy
    * std_mean_abs_diff         — |mean(y)-mean(yd)| / std(y)        (reg)
    * ks_stat                   — KS two-sample stat                 (reg)
    * wasserstein               — 1-Wasserstein distance             (reg)
    * class_prop_l1             — 0.5*sum(|p_y - p_yd|)              (cls)

  Row-wise correlation
    * abs_spearman              — |Spearman(y, y_decoy)|
    * abs_pearson               — |Pearson(y, y_decoy)|              (reg) /
                                  Pearson on category codes          (cls)
    * match_rate                — accuracy(y == y_decoy)             (cls)

  Predictability parity
    * cv_true_hgb / cv_decoy_hgb / cv_gap_hgb / cv_ratio_hgb
        Reused from the manifest's `diagnostics` block (HistGradientBoosting,
        3-fold, AUC for binary / accuracy for multiclass / R^2 for regression).
    * cv_true_lgb / cv_decoy_lgb / cv_gap_lgb / cv_ratio_lgb
        Independent re-check with LightGBM (200 trees) so the parity number
        is not tied to a single learner.

Outputs (written next to this script):
    decoy_quality.csv        — one row per task
    decoy_quality_summary.csv — Mean/Median/p90/Max per metric

This script reads only the canonical data tree and never modifies it.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def encode_features(train_df: pd.DataFrame, feat_cols: list[str],
                    dtype: str = "float32") -> np.ndarray:
    """Object/categorical -> integer codes; numeric -> coerced numeric.

    Mirrors `encode_train_for_model` in step_3b so CV diagnostics are
    comparable to those stored in the manifest.
    """
    cols = []
    for c in feat_cols:
        s = train_df[c]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            cols.append(pd.Categorical(s.astype("string")).codes.astype(dtype))
        else:
            cols.append(pd.to_numeric(s, errors="coerce").astype(dtype).values)
    return np.column_stack(cols)


def encode_target_codes(y) -> np.ndarray:
    """Stable integer encoding for a categorical target."""
    return pd.Series(y).astype("category").cat.codes.values


def quick_cv_lgb(X: np.ndarray, y, target_type: str,
                 n_folds: int = 3, max_rows: int = 50_000,
                 seed: int = 0) -> float:
    """Independent CV with LightGBM (default 200 trees, learning_rate=0.05).

    Mirrors the metric choices in `quick_cv` from step_3b:
      binary cls -> ROC-AUC, multiclass cls -> accuracy, regression -> R^2.
    """
    import lightgbm as lgb  # local import so the rest of the script is usable
                           # without lightgbm installed

    rng = np.random.default_rng(seed)
    n = len(y)
    if n > max_rows:
        idx = rng.choice(n, max_rows, replace=False)
        X = X[idx]
        y = np.asarray(y)[idx]
    else:
        y = np.asarray(y)

    if target_type == "classification":
        y_enc = encode_target_codes(y)
        n_classes = int(pd.Series(y).nunique())
        is_binary = n_classes == 2
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, te in skf.split(X, y_enc):
            m = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                random_state=seed, verbose=-1, n_jobs=1,
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

    y = np.asarray(y, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        m = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbose=-1, n_jobs=1,
        )
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.nanmean(scores))


# ---------------------------------------------------------------------------
# per-task diagnostics
# ---------------------------------------------------------------------------

def diagnose_task(task_dir: Path, run_lgb: bool, max_cv_rows: int,
                  seed: int, max_train_rows: int | None = None) -> dict:
    manifest = json.loads((task_dir / "_manifest.json").read_text())
    truth_col = manifest["true_target_column"]
    decoy_col = manifest["decoy_column"]
    target_type = manifest["target_type"]
    feat_cols = manifest["anon_feature_columns"]
    idcol = manifest["id_column"]

    read_kwargs: dict = {"low_memory": False}
    if max_train_rows is not None:
        read_kwargs["nrows"] = max_train_rows
    train = pd.read_csv(task_dir / "train.csv", **read_kwargs)
    y = train[truth_col].values
    yd = train[decoy_col].values

    out: dict = {
        "task": manifest["task"],
        "target_type": target_type,
        "n_train": int(len(train)),
        "n_features": int(len(feat_cols)),
        "truth_col": truth_col,
    }

    # --- distributional similarity ---------------------------------------
    if target_type == "regression":
        y_f = np.asarray(y, dtype=float)
        yd_f = np.asarray(yd, dtype=float)
        out["marginal_match_exact"] = bool(
            np.allclose(np.sort(y_f), np.sort(yd_f))
        )
        sd = float(np.std(y_f)) + 1e-12
        out["std_mean_abs_diff"] = float(abs(y_f.mean() - yd_f.mean()) / sd)
        try:
            out["ks_stat"] = float(stats.ks_2samp(y_f, yd_f).statistic)
        except Exception:
            out["ks_stat"] = float("nan")
        try:
            out["wasserstein"] = float(stats.wasserstein_distance(y_f, yd_f))
        except Exception:
            out["wasserstein"] = float("nan")
        out["class_prop_l1"] = float("nan")
    else:
        a = pd.Series(y).value_counts(normalize=True).sort_index()
        b = pd.Series(yd).value_counts(normalize=True).sort_index()
        union = a.index.union(b.index)
        a = a.reindex(union, fill_value=0.0)
        b = b.reindex(union, fill_value=0.0)
        out["marginal_match_exact"] = bool(
            pd.Series(y).value_counts().sort_index().equals(
                pd.Series(yd).value_counts().sort_index()
            )
        )
        out["class_prop_l1"] = float(0.5 * (a - b).abs().sum())
        out["std_mean_abs_diff"] = float("nan")
        out["ks_stat"] = float("nan")
        out["wasserstein"] = float("nan")

    # --- row-wise correlation --------------------------------------------
    if target_type == "regression":
        y_f = np.asarray(y, dtype=float)
        yd_f = np.asarray(yd, dtype=float)
        try:
            r = stats.pearsonr(y_f, yd_f).statistic
            out["abs_pearson"] = float(abs(r)) if not np.isnan(r) else float("nan")
        except Exception:
            out["abs_pearson"] = float("nan")
        try:
            rho = stats.spearmanr(y_f, yd_f).correlation
            out["abs_spearman"] = float(abs(rho)) if not np.isnan(rho) else float("nan")
        except Exception:
            out["abs_spearman"] = float("nan")
        out["match_rate"] = float("nan")
    else:
        y_c = encode_target_codes(y)
        yd_c = encode_target_codes(yd)
        # Pearson/Spearman on integer codes; meaningful for binary, only a
        # rank-order proxy for nominal multiclass (flagged via target_type).
        try:
            r = stats.pearsonr(y_c, yd_c).statistic
            out["abs_pearson"] = float(abs(r)) if not np.isnan(r) else float("nan")
        except Exception:
            out["abs_pearson"] = float("nan")
        try:
            rho = stats.spearmanr(y_c, yd_c).correlation
            out["abs_spearman"] = float(abs(rho)) if not np.isnan(rho) else float("nan")
        except Exception:
            out["abs_spearman"] = float("nan")
        out["match_rate"] = float((np.asarray(y) == np.asarray(yd)).mean())

    # --- predictability parity (HGB, from manifest) ----------------------
    diag = manifest.get("diagnostics", {}) or {}
    cv_true = diag.get("cv_true")
    cv_decoy = diag.get("cv_decoy")
    out["cv_true_hgb"] = float(cv_true) if cv_true is not None else float("nan")
    out["cv_decoy_hgb"] = float(cv_decoy) if cv_decoy is not None else float("nan")
    if cv_true is not None and cv_decoy is not None:
        out["cv_gap_hgb"] = float(abs(cv_true - cv_decoy))
        out["cv_ratio_hgb"] = (
            float(cv_decoy / cv_true) if cv_true not in (0, None) else float("nan")
        )
    else:
        out["cv_gap_hgb"] = float("nan")
        out["cv_ratio_hgb"] = float("nan")

    # --- predictability parity (LightGBM, recomputed) --------------------
    if run_lgb:
        X = encode_features(train, feat_cols)
        try:
            cv_t = quick_cv_lgb(X, y, target_type, max_rows=max_cv_rows, seed=seed)
            cv_d = quick_cv_lgb(X, yd, target_type, max_rows=max_cv_rows, seed=seed)
        except Exception as e:
            print(f"  [lgb] failed for {manifest['task']}: {e}")
            cv_t = float("nan")
            cv_d = float("nan")
        out["cv_true_lgb"] = cv_t
        out["cv_decoy_lgb"] = cv_d
        out["cv_gap_lgb"] = (
            float(abs(cv_t - cv_d))
            if not (np.isnan(cv_t) or np.isnan(cv_d)) else float("nan")
        )
        out["cv_ratio_lgb"] = (
            float(cv_d / cv_t) if cv_t not in (0,) and not np.isnan(cv_t) else float("nan")
        )
    else:
        out["cv_true_lgb"] = float("nan")
        out["cv_decoy_lgb"] = float("nan")
        out["cv_gap_lgb"] = float("nan")
        out["cv_ratio_lgb"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    # (column, scope: 'all'|'reg'|'cls', label)
    ("abs_spearman",       "all", "|Spearman(y, y_decoy)|"),
    ("abs_pearson",        "all", "|Pearson(y, y_decoy)|"),
    ("ks_stat",            "reg", "KS statistic (regression)"),
    ("wasserstein",        "reg", "Wasserstein distance (regression)"),
    ("std_mean_abs_diff",  "reg", "Std. mean abs. diff (regression)"),
    ("class_prop_l1",      "cls", "Class-proportion L1 (classification)"),
    ("match_rate",         "cls", "Match rate y==y_decoy (classification)"),
    ("cv_gap_hgb",         "all", "|CV(y) - CV(y_decoy)| (HistGB)"),
    ("cv_ratio_hgb",       "all", "CV(y_decoy)/CV(y) (HistGB)"),
    ("cv_gap_lgb",         "all", "|CV(y) - CV(y_decoy)| (LightGBM)"),
    ("cv_ratio_lgb",       "all", "CV(y_decoy)/CV(y) (LightGBM)"),
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, scope, label in SUMMARY_METRICS:
        if scope == "reg":
            sub = df[df["target_type"] == "regression"][col]
        elif scope == "cls":
            sub = df[df["target_type"] == "classification"][col]
        else:
            sub = df[col]
        sub = sub.dropna()
        if sub.empty:
            rows.append({"metric": label, "n": 0,
                         "mean": np.nan, "median": np.nan,
                         "p90": np.nan, "max": np.nan})
            continue
        rows.append({
            "metric": label,
            "n": int(len(sub)),
            "mean": float(sub.mean()),
            "median": float(sub.median()),
            "p90": float(sub.quantile(0.9)),
            "max": float(sub.max()),
        })
    # marginal-match exact rate (single-row summary)
    n_total = len(df)
    n_match = int(df["marginal_match_exact"].sum())
    rows.append({
        "metric": "Exact marginal match rate (all tasks)",
        "n": n_total,
        "mean": (n_match / n_total) if n_total else np.nan,
        "median": np.nan, "p90": np.nan, "max": np.nan,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--benchmark-dir", "--benchmark_dir", dest="benchmark_dir", type=Path,
        help="Benchmark dir produced by step_1_setup_benchmark.py. Reads "
             "<bench>/data/<slug>/ambig/{train.csv, _manifest.json}.",
    )
    p.add_argument(
        "--data_root", default=None,
        help="(Advanced) Root with per-task subdirs <root>/<slug>/{train.csv,"
             "_manifest.json}. Use this if you have a custom layout; otherwise "
             "prefer --benchmark-dir.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Where to write decoy_quality.csv. Defaults to "
             "<benchmark-dir>/audits/decoy_quality/ when --benchmark-dir is set.",
    )
    p.add_argument("--max_cv_rows", type=int, default=50_000)
    p.add_argument("--max_train_rows", type=int, default=500_000,
                   help="Cap rows read from train.csv to avoid OOM on huge "
                        "tasks (matches step_3b's --max_train_rows). The "
                        "decoy is row-aligned with train so subsampling "
                        "the first N rows preserves the (y, y_decoy) pairing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_lgb", action="store_true",
                   help="Skip the LightGBM second-pass CV (faster).")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of task names.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.benchmark_dir:
        bench = args.benchmark_dir.resolve()
        if not (bench / "data").exists():
            sys.exit(f"--benchmark-dir {bench} has no data/ subdir; "
                     "run step_1_setup_benchmark.py first.")
        # Iterate <bench>/data/<slug>/ambig/{train.csv, _manifest.json}
        task_dirs = {d.name: d / "ambig"
                     for d in (bench / "data").iterdir()
                     if d.is_dir() and (d / "ambig" / "_manifest.json").exists()}
        out_dir = Path(args.out_dir) if args.out_dir else (
            bench / "audits" / "decoy_quality")
    elif args.data_root:
        data_root = Path(args.data_root)
        task_dirs = {d.name: d for d in data_root.iterdir()
                     if d.is_dir() and (d / "_manifest.json").exists()}
        out_dir = Path(args.out_dir) if args.out_dir else (data_root.parent / "decoy_analysis")
    else:
        sys.exit("Pass either --benchmark-dir <bench> or --data_root <root>.")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(task_dirs)
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        tasks = [t for t in tasks if t in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit("No tasks matched.")

    out_csv = out_dir / "decoy_quality.csv"
    # Resume: skip tasks already present in the existing CSV (so a re-run
    # after an OOM picks up where we left off without recomputing).
    done: set[str] = set()
    if out_csv.exists():
        try:
            done = set(pd.read_csv(out_csv)["task"].astype(str).tolist())
        except Exception:
            done = set()

    print(f"Diagnosing {len(tasks)} tasks "
          f"({len(done)} already in {out_csv.name})")
    rows = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        elapsed = time.time() - t0
        if task in done:
            print(f"[{i}/{len(tasks)}] {task}  SKIP (already done)", flush=True)
            continue
        print(f"[{i}/{len(tasks)}] {task}  (elapsed {elapsed:.0f}s)", flush=True)
        try:
            row = diagnose_task(
                task_dirs[task],
                run_lgb=not args.no_lgb,
                max_cv_rows=args.max_cv_rows,
                max_train_rows=args.max_train_rows,
                seed=args.seed,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            row = {"task": task, "error": str(e)}
        rows.append(row)
        # Incremental append so we never lose progress if killed.
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False,
        )

    df = pd.read_csv(out_csv)
    print(f"\nWrote {out_csv} ({len(df)} rows)")

    summary = summarize(df.dropna(subset=["target_type"]))
    out_summary = out_dir / "decoy_quality_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {out_summary}")
    print()
    with pd.option_context("display.max_rows", None,
                           "display.max_colwidth", 60,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
