#!/usr/bin/env python3
"""Normalize raw MLE-bench scores to [0, 1] for paper-style aggregation.

For each task we already have:
  - the raw competition score (in the source metric, lower- or higher-better)
  - the per-competition Kaggle leaderboard CSV that mle-bench ships (Git LFS)

We rescale the raw score to a leaderboard rank-percentile in [0, 1] where
**1.0 = beats every team on the public leaderboard** and **0.0 = beats none**.
This is direction-safe (uses `is_lower_better` from the grader), bounded by
construction, and matches the implicit semantics of the paper's `S_full`,
`S_ambig`, `S_ask` ("Task success ... normalized to [0,1]").

Usage:
    # Normalize one results dir
    python normalize_scores.py --benchmark-dir ./benchmark \\
        --results-dir ./benchmark/results/opencode_gemini_3_flash_full

    # Normalize a model across all variants and emit a wide summary
    python normalize_scores.py --benchmark-dir ./benchmark \\
        --results-root ./benchmark/results \\
        --agent opencode --model gemini_3_flash \\
        --variants full,ambig_metric,ambig_metric_clarify \\
        --out ./benchmark/results/_normalized_gemini_3_flash.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _leaderboard_path(slug: str) -> Path:
    """Locate mle-bench's leaderboard CSV for a slug."""
    import mlebench
    return Path(mlebench.__file__).parent / "competitions" / slug / "leaderboard.csv"


def normalize_one(score: float | None, slug: str, is_lower_better: bool) -> float | None:
    """Map a raw score to a leaderboard rank-percentile in [0, 1].

    1.0 = strictly better than every leaderboard team
    0.0 = strictly worse than every leaderboard team
    Ties: averaged rank (so a score equal to the median maps to ~0.5).
    Returns None if score is None / non-numeric / leaderboard missing.
    """
    if not isinstance(score, (int, float)):
        return None
    lb_path = _leaderboard_path(slug)
    if not lb_path.exists():
        return None
    lb = pd.read_csv(lb_path)
    if "score" not in lb.columns or len(lb) == 0:
        return None
    teams = lb["score"].astype(float).to_numpy()
    n = len(teams)
    if is_lower_better:
        # Fraction of teams the agent ties or beats (i.e. has score <=).
        better = (teams > score).sum()
        ties = (teams == score).sum()
    else:
        better = (teams < score).sum()
        ties = (teams == score).sum()
    return float((better + 0.5 * ties) / n)


def annotate_results_dir(results_dir: Path) -> list[dict]:
    """Read every <slug>/_grade.json under results_dir, write S_norm into it,
    return a list of summary dicts."""
    rows: list[dict] = []
    for grade_file in sorted(results_dir.glob("*/_grade.json")):
        slug = grade_file.parent.name
        try:
            g = json.loads(grade_file.read_text())
        except Exception as e:
            rows.append({"slug": slug, "error": f"unreadable _grade.json: {e}"})
            continue
        score = g.get("score")
        ilb = g.get("is_lower_better")
        s_norm = normalize_one(score, slug, bool(ilb)) if ilb is not None else None
        g["score_norm_01"] = s_norm
        grade_file.write_text(json.dumps(g, indent=2, default=str))
        rows.append({
            "slug": slug,
            "score": score,
            "is_lower_better": ilb,
            "score_norm_01": s_norm,
            "above_median": g.get("above_median"),
            "any_medal": g.get("any_medal"),
            "valid_submission": g.get("valid_submission"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--benchmark-dir", type=Path, required=True,
                    help="Benchmark directory (only used for symmetry; leaderboards live in mlebench).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--results-dir", type=Path, default=None,
                   help="A single run directory (e.g. .../results/opencode_<model>_full).")
    g.add_argument("--results-root", type=Path, default=None,
                   help="Parent dir under which run dirs <agent>_<model>_<variant>[_clarify] live.")
    ap.add_argument("--agent", default=None, help="With --results-root: agent prefix (e.g. opencode).")
    ap.add_argument("--model", default=None, help="With --results-root: model id (e.g. gemini_3_flash).")
    ap.add_argument("--variants", default="full,ambig_metric",
                    help="With --results-root: comma-separated variant names. Append '_clarify' for the Step 3 run.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV path for the wide summary (only with --results-root).")
    args = ap.parse_args()

    if args.results_dir is not None:
        rows = annotate_results_dir(args.results_dir.resolve())
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"No _grade.json under {args.results_dir}")
            return
        print(df.to_string(index=False))
        if "score_norm_01" in df:
            print(f"\nMean S_norm = {df['score_norm_01'].mean():.4f}  (n={df['score_norm_01'].notna().sum()})")
        return

    # --results-root mode: gather all variants for one agent+model into a wide table.
    if not (args.agent and args.model):
        sys.exit("--results-root requires --agent and --model.")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    per_variant: dict[str, pd.DataFrame] = {}
    for v in variants:
        run_dir = args.results_root / f"{args.agent}_{args.model}_{v}"
        if not run_dir.exists():
            print(f"  (skip) missing {run_dir}", file=sys.stderr)
            continue
        rows = annotate_results_dir(run_dir)
        if not rows:
            continue
        per_variant[v] = pd.DataFrame(rows)[["slug", "score", "score_norm_01"]] \
            .rename(columns={"score": f"score_{v}", "score_norm_01": f"S_{v}"})

    if not per_variant:
        sys.exit("No results found.")

    wide = None
    for v, df in per_variant.items():
        wide = df if wide is None else wide.merge(df, on="slug", how="outer")
    # Paper-style deltas
    if "S_full" in wide and "S_ambig_metric" in wide:
        wide["delta_ambig"] = wide["S_ambig_metric"] - wide["S_full"]
    if "S_ambig_metric" in wide and "S_ambig_metric_clarify" in wide:
        wide["delta_ask"] = wide["S_ambig_metric_clarify"] - wide["S_ambig_metric"]

    print(wide.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    means = {c: wide[c].mean() for c in wide.columns if c.startswith(("S_", "delta_"))}
    print("\nMeans (macro-averaged across tasks):")
    for k, v in means.items():
        print(f"  {k:25s} {v: .4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
