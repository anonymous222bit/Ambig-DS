#!/usr/bin/env python3
"""Step V3 — noise/swap-rate sensitivity ablation.

Regenerates the target-ambiguity decoys at several noise/swap rates
(`--noise_levels`, percentages) and recomputes the V1 quality diagnostics
plus a single inferability selector (Selector B from V2: pick the column
with the higher quick-CV signal).

For each noise level L (e.g. 0, 5, 10, 20):

  1. Calls `step_3b_strong_decoy_generator.py` for every task in the
     supplied --tasks_csvs into a fresh directory
        <out_root>/ablation/n{LL}/
     so the canonical `target_ambig/data/` tree is **never** touched.

  2. Walks the regenerated tree and reads each task's `_manifest.json` +
     `train.csv` to compute, for each task:
        * marginal_match_exact      (manifest)
        * |Spearman(y, y_decoy)|    (recomputed from train.csv)
        * KS statistic              (regression only)
        * class_prop_l1             (classification only)
        * cv_true / cv_decoy        (manifest)
        * cv_gap = |cv_true - cv_decoy|
        * selector_B_correct        (1 if pick by higher-CV equals truth)

  3. Aggregates per-level statistics (Mean / Median / Max for the headline
     metrics, accuracy + 95% Wilson CI for Selector B).

Outputs (in --out_dir):
    ablation/n{LL}/...                   — regenerated decoys per level
    ablation_per_task.csv                — every (level, task) row
    ablation_summary.csv                 — per-level aggregates
    tex/decoy_ablation_table.tex         — paper-ready table

The 10% column reproduces the V1 / V2 numbers and is used as the reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(os.environ.get("AMBIG_DSBENCH_ROOT", Path.cwd())).resolve()
STEP_3B = Path(__file__).resolve().parent / "step_3b_v2_calibrated_decoy.py"
DEFAULT_SRC = (REPO / "Dataset" / "data_modeling" / "data" / "data"
               / "data_resplit")


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

def combined_tasks_csv(tasks_csvs: list[Path], dest: Path) -> Path:
    """Concatenate one or more pilot CSVs into a single CSV the regen
    script can consume."""
    frames = [pd.read_csv(p) for p in tasks_csvs]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["task"]).reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


# ---------------------------------------------------------------------------
# regen
# ---------------------------------------------------------------------------

def regenerate(tasks_csv: Path, out_root: Path, noise_pct: float,
               src_data_root: Path, max_train_rows: int,
               only: list[str] | None) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(STEP_3B),
        "--tasks_csv", str(tasks_csv),
        "--out_root", str(out_root),
        "--src_data_root", str(src_data_root),
        "--noise_classification", str(noise_pct / 100.0),
        "--noise_regression", str(noise_pct / 100.0),
        "--max_train_rows", str(max_train_rows),
    ]
    if only:
        cmd += ["--only", ",".join(only)]
    print(f"  -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# per-task diagnostics (lightweight; reuses logic from V1 selector B)
# ---------------------------------------------------------------------------

def task_diagnostics(task_dir: Path, max_rows: int) -> dict:
    manifest = json.loads((task_dir / "_manifest.json").read_text())
    truth_col = manifest["true_target_column"]
    decoy_col = manifest["decoy_column"]
    target_type = manifest["target_type"]

    train = pd.read_csv(task_dir / "train.csv", nrows=max_rows,
                        low_memory=False)
    y = train[truth_col].values
    yd = train[decoy_col].values

    diag = manifest.get("diagnostics", {}) or {}
    cv_t = diag.get("cv_true")
    cv_d = diag.get("cv_decoy")
    out: dict = {
        "task": manifest["task"],
        "target_type": target_type,
        "n_train": int(len(train)),
        "marginal_match_exact": bool(diag.get("marginal_match_exact", False)),
        "cv_true": float(cv_t) if cv_t is not None else float("nan"),
        "cv_decoy": float(cv_d) if cv_d is not None else float("nan"),
    }
    out["cv_gap"] = (
        float(abs(cv_t - cv_d))
        if cv_t is not None and cv_d is not None else float("nan")
    )

    # |Spearman(y, y_decoy)|
    if target_type == "regression":
        try:
            rho = stats.spearmanr(np.asarray(y, dtype=float),
                                  np.asarray(yd, dtype=float)).correlation
            out["abs_spearman"] = (float(abs(rho))
                                   if rho is not None and not np.isnan(rho)
                                   else float("nan"))
        except Exception:
            out["abs_spearman"] = float("nan")
        # KS
        try:
            out["ks_stat"] = float(stats.ks_2samp(
                np.asarray(y, dtype=float),
                np.asarray(yd, dtype=float),
            ).statistic)
        except Exception:
            out["ks_stat"] = float("nan")
        out["class_prop_l1"] = float("nan")
    else:
        y_c = pd.Series(y).astype("category").cat.codes.values
        yd_c = pd.Series(yd).astype("category").cat.codes.values
        try:
            rho = stats.spearmanr(y_c, yd_c).correlation
            out["abs_spearman"] = (float(abs(rho))
                                   if rho is not None and not np.isnan(rho)
                                   else float("nan"))
        except Exception:
            out["abs_spearman"] = float("nan")
        out["ks_stat"] = float("nan")
        # class-proportion L1
        a = pd.Series(y).value_counts(normalize=True).sort_index()
        b = pd.Series(yd).value_counts(normalize=True).sort_index()
        union = a.index.union(b.index)
        a = a.reindex(union, fill_value=0.0)
        b = b.reindex(union, fill_value=0.0)
        out["class_prop_l1"] = float(0.5 * (a - b).abs().sum())

    # Selector B: pick column with higher quick-CV signal.
    if cv_t is not None and cv_d is not None:
        cv_v1 = cv_t if truth_col == "val_1" else cv_d
        cv_v2 = cv_d if truth_col == "val_1" else cv_t
        pick = "val_1" if cv_v1 >= cv_v2 else "val_2"
        out["selector_B_pick"] = pick
        out["selector_B_correct"] = int(pick == truth_col)
    else:
        out["selector_B_pick"] = None
        out["selector_B_correct"] = None

    return out


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


def summarize_level(df: pd.DataFrame, noise_pct: float) -> dict:
    n = len(df)
    sel = df["selector_B_correct"].dropna().astype(int)
    acc, lo, hi = wilson_ci(int(sel.sum()), int(len(sel))) if len(sel) else (
        float("nan"), float("nan"), float("nan")
    )
    return {
        "noise_pct":           noise_pct,
        "n_tasks":             n,
        "marginal_exact_pct":  100.0 * float(df["marginal_match_exact"].mean())
                                if n else float("nan"),
        "abs_spearman_mean":   float(df["abs_spearman"].mean(skipna=True)),
        "abs_spearman_median": float(df["abs_spearman"].median(skipna=True)),
        "abs_spearman_max":    float(df["abs_spearman"].max(skipna=True)),
        "cv_gap_mean":         float(df["cv_gap"].mean(skipna=True)),
        "cv_gap_median":       float(df["cv_gap"].median(skipna=True)),
        "selB_accuracy":       acc,
        "selB_ci_low":         lo,
        "selB_ci_high":        hi,
        "selB_n":              int(len(sel)),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out_dir",
                   default=str(REPO / "final_data_v3" / "target_ambig"
                               / "decoy_analysis"))
    p.add_argument("--ablation_root", default=None,
                   help="Where to put regenerated decoys; defaults to "
                        "<out_dir>/ablation.")
    p.add_argument("--tasks_csvs", nargs="+",
                   default=[
                       str(REPO / "final_data_v3" / "target_ambig"
                           / "target_ambiguity_tasks.csv"),
                       str(REPO / "final_data_v3" / "target_ambig"
                           / "expansion_tasks.csv"),
                   ],
                   help="One or more pilot CSVs to regenerate.")
    p.add_argument("--src_data_root", default=str(DEFAULT_SRC))
    p.add_argument("--noise_levels",
                   default="0,5,10,20",
                   help="Comma-separated percentages.")
    p.add_argument("--max_train_rows", type=int, default=200_000)
    p.add_argument("--max_diag_rows", type=int, default=200_000,
                   help="Cap rows when reading regenerated train.csv for "
                        "Spearman / KS / class-prop computation.")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of task names (passed to "
                        "step_3b's --only and used to filter diagnostics).")
    p.add_argument("--skip_regen", action="store_true",
                   help="Don't call step_3b; assume <ablation_root>/n{LL}/ "
                        "is already populated. Useful for re-aggregating.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tex").mkdir(exist_ok=True)
    abl_root = Path(args.ablation_root) if args.ablation_root else (
        out_dir / "ablation"
    )
    abl_root.mkdir(parents=True, exist_ok=True)

    levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    only_list = [t.strip() for t in args.only.split(",")] if args.only else None
    src = Path(args.src_data_root)

    # Combined tasks CSV (so step_3b processes pilot + expansion in one call).
    combined = combined_tasks_csv(
        [Path(x) for x in args.tasks_csvs],
        abl_root / "_combined_tasks.csv",
    )
    print(f"Combined {len(args.tasks_csvs)} CSVs -> {combined} "
          f"({len(pd.read_csv(combined))} tasks)")

    per_task_rows: list[dict] = []
    summary_rows: list[dict] = []
    t0 = time.time()
    for lvl in levels:
        lvl_label = f"n{int(round(lvl)):02d}"
        lvl_root = abl_root / lvl_label
        elapsed = time.time() - t0
        print(f"\n=== noise = {lvl}% (-> {lvl_root})  elapsed {elapsed:.0f}s ===",
              flush=True)
        if not args.skip_regen:
            try:
                regenerate(combined, lvl_root, lvl, src,
                           args.max_train_rows, only_list)
            except subprocess.CalledProcessError as e:
                print(f"  REGEN FAILED at noise={lvl}%: {e}")
                # fall through and aggregate whatever was completed
        # diagnostics
        task_dirs = sorted(d for d in (lvl_root / "data").iterdir()
                           if d.is_dir() and (d / "_manifest.json").exists())
        if only_list:
            keep = set(only_list)
            task_dirs = [d for d in task_dirs if d.name in keep]
        print(f"  diagnostics on {len(task_dirs)} tasks", flush=True)
        rows = []
        for td in task_dirs:
            try:
                r = task_diagnostics(td, max_rows=args.max_diag_rows)
            except Exception as e:
                print(f"    {td.name}: diag failed {e}")
                continue
            r["noise_pct"] = lvl
            rows.append(r)
        if not rows:
            print(f"  no tasks succeeded at noise={lvl}%; skipping")
            continue
        df_lvl = pd.DataFrame(rows)
        per_task_rows.extend(rows)
        summary_rows.append(summarize_level(df_lvl, lvl))

    if not per_task_rows:
        sys.exit("No diagnostics computed — nothing to write.")

    per_task_csv = out_dir / "ablation_per_task.csv"
    pd.DataFrame(per_task_rows).to_csv(per_task_csv, index=False)
    print(f"\nWrote {per_task_csv}")

    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # LaTeX
    def fmt(x, pct=False):
        if pd.isna(x):
            return "--"
        return f"{100 * x:.1f}\\%" if pct else f"{x:.3f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of decoy diagnostics to construction noise. "
        r"Decoys are regenerated for every retained target-ambiguity task at "
        r"the listed swap (classification) / jitter (regression) rates and "
        r"the resulting diagnostics are aggregated. The retained benchmark "
        r"setting is 10\%. Selector~B is the basic CV heuristic from "
        r"Table~\ref{tab:target_inferability_audit} (pick the column with "
        r"the higher quick-CV signal); accuracy near or below chance "
        r"indicates the rate is not a load-bearing hyperparameter.}",
        r"\label{tab:decoy_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Noise rate & Tasks & "
        r"$|\rho(y,y_{\mathrm{decoy}})|$ (med.) & "
        r"$|$CV gap$|$ (med.) & "
        r"Marginal exact & "
        r"Selector~B accuracy \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{int(round(r['noise_pct']))}\\% & "
            f"{int(r['n_tasks'])} & "
            f"{fmt(r['abs_spearman_median'])} & "
            f"{fmt(r['cv_gap_median'])} & "
            f"{r['marginal_exact_pct']:.0f}\\% & "
            f"{fmt(r['selB_accuracy'], pct=True)} "
            f"({fmt(r['selB_ci_low'], pct=True)}, "
            f"{fmt(r['selB_ci_high'], pct=True)})"
            r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = out_dir / "tex" / "decoy_ablation_table.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {tex_path}")

    print()
    with pd.option_context("display.max_colwidth", 80,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
