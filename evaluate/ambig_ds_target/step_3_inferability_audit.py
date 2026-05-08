#!/usr/bin/env python3
"""Step V2 — target-inferability audit for the target-ambiguity benchmark.

For every task in the benchmark dir produced by step_1_setup_benchmark.py:
    <benchmark-dir>/data/<slug>/ambig/{train.csv, _manifest.json}
runs four selectors that try to identify which of `val_1` / `val_2` is the
intended target, using only the ambiguous prompt + data observation.
Each selector emits a single bit (the column it picks). We compare to the
truth recorded in `_manifest.json` and aggregate accuracy across tasks.

Selectors
---------
A — Marginal-entropy heuristic
    Pick the candidate with **lower** marginal entropy. Real Kaggle targets
    tend to be more skewed than rank-mapped decoys; if the heuristic is
    correct in spirit, this should peek above chance. We expect near-50%.

B — Basic CV heuristic (HistGradientBoosting)
    Pick the candidate with the **higher** 3-fold CV signal. This is the
    exact attack the reviewer worries about. Reuses the per-column CV
    numbers already stored in `_manifest.json['diagnostics']`.

C — Strong AutoML CV heuristic (LightGBM, 500 trees, 5-fold)
    Same as B but with a deeper learner and 5-fold CV. Tests whether a
    stronger feature-selection / CV routine could break the construction.

D — LLM schema/prompt heuristic
    Send the original task prompt + the train schema (column names) +
    20 sampled rows + the sample-submission stub to one cheap model and
    ask it to pick `val_1` or `val_2`. Repeated `--llm_seeds` times per
    task with non-zero temperature; majority vote per task. Off by default
    (`--skip_llm`) so the cheap selectors can be iterated quickly.

E — LLM no-prompt heuristic
    Same as D but **without** the original Kaggle prompt (only schema +
    sample rows + submission stub). Tests whether the decoy construction
    itself leaks the truth, independently of any semantic information the
    Kaggle prompt may carry about the dataset.

Outputs (in --out_dir):
    inferability_audit.csv       — one row per (task, selector) with the
                                    pick + correctness flag + raw scores.
    inferability_audit_summary.csv — accuracy + Wilson 95% CI per selector.
    tex/target_inferability_table.tex — paper-ready table.

Reads only the canonical data tree and never modifies it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# helpers reused from V1 (kept local so this script is standalone)
# ---------------------------------------------------------------------------

def encode_features(train_df: pd.DataFrame, feat_cols: list[str],
                    dtype: str = "float32") -> np.ndarray:
    cols = []
    for c in feat_cols:
        s = train_df[c]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            cols.append(pd.Categorical(s.astype("string")).codes.astype(dtype))
        else:
            cols.append(pd.to_numeric(s, errors="coerce").astype(dtype).values)
    return np.column_stack(cols)


def encode_target_codes(y) -> np.ndarray:
    return pd.Series(y).astype("category").cat.codes.values


# ---------------------------------------------------------------------------
# selector A — marginal entropy
# ---------------------------------------------------------------------------

def shannon_entropy(values, bins: int = 32) -> float:
    """Entropy proxy that works for both categorical and continuous columns.

    Categorical / int with few unique values  -> Shannon entropy of empirical
        probabilities.
    Otherwise                                  -> histogram-based differential
        entropy approximation (in nats).
    """
    s = pd.Series(values).dropna()
    n_unique = s.nunique()
    if n_unique <= max(20, int(0.01 * len(s))):
        p = s.value_counts(normalize=True).values
        return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
    # continuous: histogram
    arr = np.asarray(s, dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    arr = arr[np.isfinite(arr)]
    counts, edges = np.histogram(arr, bins=bins)
    p = counts / counts.sum() if counts.sum() else counts
    width = edges[1] - edges[0] if len(edges) > 1 else 1.0
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz] / max(width, 1e-12))).sum())


def selector_marginal_entropy(train: pd.DataFrame) -> tuple[str, dict]:
    h1 = shannon_entropy(train["val_1"])
    h2 = shannon_entropy(train["val_2"])
    pick = "val_1" if h1 <= h2 else "val_2"
    return pick, {"entropy_val_1": h1, "entropy_val_2": h2}


# ---------------------------------------------------------------------------
# selector B — CV from manifest
# ---------------------------------------------------------------------------

def selector_basic_cv(manifest: dict) -> tuple[str | None, dict]:
    diag = manifest.get("diagnostics", {}) or {}
    cv_t = diag.get("cv_true")
    cv_d = diag.get("cv_decoy")
    truth = manifest["true_target_column"]
    decoy = manifest["decoy_column"]
    if cv_t is None or cv_d is None:
        return None, {"cv_val_1": None, "cv_val_2": None}
    cv_v1 = cv_t if truth == "val_1" else cv_d
    cv_v2 = cv_d if truth == "val_1" else cv_t
    pick = "val_1" if cv_v1 >= cv_v2 else "val_2"
    return pick, {"cv_val_1": float(cv_v1), "cv_val_2": float(cv_v2)}


# ---------------------------------------------------------------------------
# selector C — strong LightGBM CV
# ---------------------------------------------------------------------------

def strong_cv_lgb(X: np.ndarray, y, target_type: str,
                  n_folds: int = 3, max_rows: int = 30_000,
                  seed: int = 0) -> float:
    import lightgbm as lgb
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
        is_binary = int(pd.Series(y).nunique()) == 2
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, te in skf.split(X, y_enc):
            m = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=63,
                min_data_in_leaf=20, feature_fraction=0.9,
                random_state=seed, verbose=-1, n_jobs=-1,
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
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=20, feature_fraction=0.9,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.nanmean(scores))


def selector_strong_cv(train: pd.DataFrame, manifest: dict,
                       max_cv_rows: int, seed: int) -> tuple[str, dict]:
    feat_cols = manifest["anon_feature_columns"]
    target_type = manifest["target_type"]
    X = encode_features(train, feat_cols)
    cv_v1 = strong_cv_lgb(X, train["val_1"].values, target_type,
                          max_rows=max_cv_rows, seed=seed)
    cv_v2 = strong_cv_lgb(X, train["val_2"].values, target_type,
                          max_rows=max_cv_rows, seed=seed)
    pick = "val_1" if (np.nan_to_num(cv_v1, nan=-np.inf)
                       >= np.nan_to_num(cv_v2, nan=-np.inf)) else "val_2"
    return pick, {"strong_cv_val_1": float(cv_v1), "strong_cv_val_2": float(cv_v2)}


# ---------------------------------------------------------------------------
# selector D — LLM schema heuristic
# ---------------------------------------------------------------------------

LLM_SYS = (
    "You are an expert data scientist auditing a tabular machine-learning "
    "task. The task description below names a real target semantically, but "
    "the training data has TWO candidate target columns named 'val_1' and "
    "'val_2'. Exactly one of them is the intended target; the other is a "
    "decoy with similar marginal distribution. Your job is to decide which "
    "of 'val_1' and 'val_2' is the intended target.\n\n"
    "Reply with a single JSON object on one line: "
    '{\"pick\": \"val_1\"} or {\"pick\": \"val_2\"}, plus an optional '
    '\"reason\" field of at most one short sentence. No markdown, no '
    "backticks."
)


def llm_user_message(prompt_text: str, schema_cols: list[str],
                     sample_rows_csv: str, submission_head: str,
                     include_prompt: bool = True) -> str:
    parts: list[str] = []
    if include_prompt:
        parts.append(
            "TASK PROMPT:\n"
            "----------------\n"
            f"{prompt_text.strip()}\n"
            "----------------\n"
        )
    else:
        parts.append(
            "NOTE: The original task prompt is intentionally withheld for "
            "this audit. Decide using only the schema and data below.\n"
        )
    parts.append(f"TRAIN SCHEMA (column order): {', '.join(schema_cols)}\n")
    parts.append(
        "TRAIN SAMPLE (first 20 rows, CSV):\n"
        f"{sample_rows_csv}\n"
    )
    parts.append(
        "SAMPLE SUBMISSION (first 5 rows, CSV):\n"
        f"{submission_head}\n"
    )
    parts.append(
        "Which column ('val_1' or 'val_2') is the intended target? "
        "Reply ONLY with the JSON object."
    )
    return "\n".join(parts)


def parse_llm_pick(text: str) -> str | None:
    import re
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    try:
        obj = json.loads(t)
        p = str(obj.get("pick", "")).strip()
        if p in ("val_1", "val_2"):
            return p
    except Exception:
        pass
    m = re.search(r"\bval_[12]\b", t)
    return m.group(0) if m else None


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    here = Path(__file__).resolve()
    # Fall back to a sibling .env file (helpful when running from a workspace
    # checkout). Otherwise raise.
    for env_path in [here.parents[3] / ".env",
                     here.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("No OPENAI_API_KEY available")


def selector_llm(client, llm_model: str, prompt_text: str, train: pd.DataFrame,
                 submission_df: pd.DataFrame, n_seeds: int,
                 include_prompt: bool = True,
                 max_tokens: int = 200) -> tuple[str | None, dict]:
    """Returns (majority_pick, info). info contains per-seed picks + raw text."""
    schema = list(train.columns)
    sample_rows = train.head(20).to_csv(index=False)
    sub_head = submission_df.head(5).to_csv(index=False)
    user_msg = llm_user_message(
        prompt_text, schema, sample_rows, sub_head,
        include_prompt=include_prompt,
    )

    picks: list[str | None] = []
    raws: list[str] = []
    for s in range(n_seeds):
        for attempt in range(3):
            try:
                # Some Bedrock-hosted models reject `seed` and/or
                # `temperature`. We omit both and rely on the provider's
                # default sampling; the n_seeds repeats still give us
                # multiple draws via majority vote.
                r = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": LLM_SYS},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=max_tokens,
                )
                text = (r.choices[0].message.content or "").strip()
                break
            except Exception as e:
                err = str(e)
                if attempt == 2:
                    text = f"<<error: {err}>>"
                time.sleep(2 ** attempt)
        raws.append(text)
        picks.append(parse_llm_pick(text))

    valid = [p for p in picks if p in ("val_1", "val_2")]
    if not valid:
        return None, {"llm_picks": picks, "llm_raw": raws}
    pick = max(set(valid), key=valid.count)
    return pick, {"llm_picks": picks, "llm_raw": raws}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# per-task
# ---------------------------------------------------------------------------

def _find_sample_submission(task_dir: Path) -> Path:
    """Locate sample_submission.csv, trying the ambig dir first, then full."""
    for name in ("sample_submission.csv", "sampleSubmission.csv"):
        p = task_dir / name
        if p.exists():
            return p
    # ambig dir may not have it; fall back to the sibling full/ dir
    full_dir = task_dir.parent / "full"
    for name in ("sample_submission.csv", "sampleSubmission.csv"):
        p = full_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No sample submission in {task_dir} or {full_dir}")


def audit_task(task_dir: Path, args, llm_client) -> list[dict]:
    manifest = json.loads((task_dir / "_manifest.json").read_text())
    truth = manifest["true_target_column"]
    target_type = manifest["target_type"]
    feat_cols = manifest["anon_feature_columns"]

    read_kwargs: dict = {"low_memory": False}
    if args.max_train_rows is not None:
        read_kwargs["nrows"] = args.max_train_rows
    train = pd.read_csv(task_dir / "train.csv", **read_kwargs)

    rows: list[dict] = []

    # -- A: marginal entropy ----------------------------------------------
    pick_a, info_a = selector_marginal_entropy(train)
    rows.append({
        "task": manifest["task"], "selector": "A_marginal_entropy",
        "pick": pick_a, "truth": truth,
        "correct": int(pick_a == truth),
        "info": json.dumps(info_a),
    })

    # -- B: basic CV (manifest) -------------------------------------------
    pick_b, info_b = selector_basic_cv(manifest)
    rows.append({
        "task": manifest["task"], "selector": "B_basic_cv_hgb",
        "pick": pick_b, "truth": truth,
        "correct": int(pick_b == truth) if pick_b else 0,
        "info": json.dumps(info_b),
    })

    # -- C: strong LightGBM CV --------------------------------------------
    if args.skip_strong:
        pick_c, info_c = None, {}
    else:
        try:
            pick_c, info_c = selector_strong_cv(
                train, manifest,
                max_cv_rows=args.max_cv_rows, seed=args.seed,
            )
        except Exception as e:
            print(f"  [C-strong] failed: {e}")
            pick_c, info_c = None, {"error": str(e)}
    rows.append({
        "task": manifest["task"], "selector": "C_strong_cv_lgb",
        "pick": pick_c, "truth": truth,
        "correct": int(pick_c == truth) if pick_c else 0,
        "info": json.dumps(info_c),
    })

    # -- D: LLM heuristic (with prompt) -----------------------------------
    if args.skip_llm or llm_client is None:
        pick_d, info_d = None, {}
    else:
        try:
            prompt_path = task_dir / "prompt.txt"
            prompt_text = prompt_path.read_text() if prompt_path.exists() else ""
            sub = pd.read_csv(_find_sample_submission(task_dir), low_memory=False)
            pick_d, info_d = selector_llm(
                llm_client, args.llm_model, prompt_text, train, sub,
                n_seeds=args.llm_seeds, include_prompt=True,
            )
        except Exception as e:
            print(f"  [D-llm] failed: {e}")
            pick_d, info_d = None, {"error": str(e)}
    rows.append({
        "task": manifest["task"], "selector": "D_llm_schema",
        "pick": pick_d, "truth": truth,
        "correct": int(pick_d == truth) if pick_d else 0,
        "info": json.dumps(info_d),
    })

    # -- E: LLM heuristic (no prompt) -------------------------------------
    if args.skip_llm or llm_client is None:
        pick_e, info_e = None, {}
    else:
        try:
            sub = pd.read_csv(_find_sample_submission(task_dir), low_memory=False)
            pick_e, info_e = selector_llm(
                llm_client, args.llm_model, "", train, sub,
                n_seeds=args.llm_seeds, include_prompt=False,
            )
        except Exception as e:
            print(f"  [E-llm-noprompt] failed: {e}")
            pick_e, info_e = None, {"error": str(e)}
    rows.append({
        "task": manifest["task"], "selector": "E_llm_noprompt",
        "pick": pick_e, "truth": truth,
        "correct": int(pick_e == truth) if pick_e else 0,
        "info": json.dumps(info_e),
    })

    return rows


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
             "_manifest.json}. Otherwise prefer --benchmark-dir.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output dir. Defaults to <benchmark-dir>/audits/inferability/ "
             "when --benchmark-dir is set.",
    )
    p.add_argument("--max_cv_rows", type=int, default=20_000)
    p.add_argument("--max_train_rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_strong", action="store_true",
                   help="Skip selector C (strong LightGBM CV).")
    p.add_argument("--skip_llm", action="store_true",
                   help="Skip selector D (LLM heuristic).")
    p.add_argument("--llm_model", default=os.environ.get("AMBIG_LLM_MODEL", "gpt-4o-mini"))
    p.add_argument("--llm_base_url",
                   default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--llm_seeds", type=int, default=3,
                   help="Number of independent LLM samples per task; "
                        "majority vote determines the pick.")
    p.add_argument("--only", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.benchmark_dir:
        bench = args.benchmark_dir.resolve()
        if not (bench / "data").exists():
            sys.exit(f"--benchmark-dir {bench} has no data/ subdir; "
                     "run step_1_setup_benchmark.py first.")
        task_dirs = {d.name: d / "ambig"
                     for d in (bench / "data").iterdir()
                     if d.is_dir() and (d / "ambig" / "_manifest.json").exists()}
        out_dir = Path(args.out_dir) if args.out_dir else (
            bench / "audits" / "inferability")
    elif args.data_root:
        data_root = Path(args.data_root)
        task_dirs = {d.name: d for d in data_root.iterdir()
                     if d.is_dir() and (d / "_manifest.json").exists()}
        out_dir = Path(args.out_dir) if args.out_dir else (data_root.parent / "decoy_analysis")
    else:
        sys.exit("Pass either --benchmark-dir <bench> or --data_root <root>.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tex").mkdir(exist_ok=True)
    out_csv = out_dir / "inferability_audit.csv"

    tasks = sorted(task_dirs)
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        tasks = [t for t in tasks if t in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit("No tasks matched.")

    # resume support. Only count rows where the selector actually produced a
    # non-null pick as "done"; failed rows (e.g. transient LLM errors) get
    # retried on the next run.
    done_keys: set[tuple[str, str]] = set()
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            prev = prev.drop_duplicates(
                subset=["task", "selector"], keep="last")
            ok = prev[prev["pick"].notna()]
            done_keys = {(r.task, r.selector) for r in ok.itertuples()}
        except Exception:
            done_keys = set()

    # set up LLM client lazily
    llm_client = None
    if not args.skip_llm:
        from openai import OpenAI
        llm_client = OpenAI(api_key=load_api_key(), base_url=args.llm_base_url)

    print(f"Auditing {len(tasks)} tasks")
    print(f"  selectors: A={'on'} B={'on'} "
          f"C={'on' if not args.skip_strong else 'SKIP'} "
          f"D={'on' if not args.skip_llm else 'SKIP'}")

    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        elapsed = time.time() - t0
        # Decide whether all 4 selectors are already cached for this task.
        needed_selectors = {"A_marginal_entropy", "B_basic_cv_hgb"}
        if not args.skip_strong:
            needed_selectors.add("C_strong_cv_lgb")
        if not args.skip_llm:
            needed_selectors.add("D_llm_schema")
            needed_selectors.add("E_llm_noprompt")
        already = {sel for (t_, sel) in done_keys if t_ == task}
        if needed_selectors.issubset(already):
            print(f"[{i}/{len(tasks)}] {task}  SKIP (cached)", flush=True)
            continue
        print(f"[{i}/{len(tasks)}] {task}  (elapsed {elapsed:.0f}s)", flush=True)
        try:
            rows = audit_task(task_dirs[task], args, llm_client)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        # Filter out rows we already have, then append new ones.
        new_rows = [r for r in rows if (r["task"], r["selector"]) not in already]
        if not new_rows:
            continue
        pd.DataFrame(new_rows).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False,
        )
        for r in new_rows:
            done_keys.add((r["task"], r["selector"]))

    if not out_csv.exists():
        print("\nNo audit rows were produced — nothing to summarize.")
        return

    df = pd.read_csv(out_csv)
    # Dedup: keep the LAST row per (task, selector) in append order. The CSV
    # is append-only, so the row at the bottom is the most recent attempt.
    n_before = len(df)
    df = (df.drop_duplicates(subset=["task", "selector"], keep="last")
            .reset_index(drop=True))
    n_after = len(df)
    if n_after < n_before:
        df.to_csv(out_csv, index=False)
        print(f"Deduped {n_before - n_after} duplicate (task, selector) rows; "
              f"rewrote {out_csv}")
    print(f"\nLoaded {n_after} (task, selector) rows from {out_csv}")

    SELECTOR_LABELS = {
        "A_marginal_entropy": "Marginal-entropy heuristic",
        "B_basic_cv_hgb":     "Basic CV heuristic (HistGradientBoosting)",
        "C_strong_cv_lgb":    "Strong AutoML CV heuristic (LightGBM, 200 trees, 3-fold)",
        "D_llm_schema":       f"LLM heuristic, with original prompt ({args.llm_model})",
        "E_llm_noprompt":     f"LLM heuristic, schema + data only ({args.llm_model})",
    }

    summary_rows = []
    for sel, label in SELECTOR_LABELS.items():
        sub = df[(df["selector"] == sel) & (df["pick"].notna())]
        if sub.empty:
            summary_rows.append({"selector": label, "n": 0,
                                 "accuracy": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan})
            continue
        n = int(len(sub))
        k = int(sub["correct"].sum())
        acc, lo, hi = wilson_ci(k, n)
        summary_rows.append({"selector": label, "n": n,
                             "accuracy": acc, "ci_low": lo, "ci_high": hi})

    # Always include "Random choice"
    summary_rows = [
        {"selector": "Random choice", "n": int(df["task"].nunique()),
         "accuracy": 0.5, "ci_low": 0.5, "ci_high": 0.5}
    ] + summary_rows

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "inferability_audit_summary.csv", index=False)
    print(f"Wrote {out_dir / 'inferability_audit_summary.csv'}")

    # LaTeX
    def fmt_pct(x):
        return "--" if pd.isna(x) else f"{100 * x:.1f}\\%"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Target inferability audit. Each selector attempts to "
        r"identify the intended target from the ambiguous prompt--data "
        r"observation without access to the source task. Accuracy near "
        r"chance indicates that the intended target is not recoverable from "
        r"simple distributional, cross-validation, or schema cues. "
        r"95\% Wilson confidence intervals shown.}",
        r"\label{tab:target_inferability_audit}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Selector & Intended-target accuracy & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        if r["selector"] == "Random choice":
            ci = "--"
        else:
            ci = (f"[{fmt_pct(r['ci_low'])}, {fmt_pct(r['ci_high'])}]"
                  if not pd.isna(r["ci_low"]) else "--")
        lines.append(
            f"{r['selector']} & {fmt_pct(r['accuracy'])} & {ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines) + "\n"
    (out_dir / "tex" / "target_inferability_table.tex").write_text(tex)
    print(f"Wrote {out_dir / 'tex' / 'target_inferability_table.tex'}")

    print()
    with pd.option_context("display.max_colwidth", 80,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
