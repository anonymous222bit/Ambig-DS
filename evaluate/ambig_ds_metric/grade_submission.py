#!/usr/bin/env python3
"""Grade a submission CSV against MLE-bench ground truth.

Standalone utility — grade a single submission or batch-grade a results directory.

Usage:
    # Grade one submission
    python grade_submission.py --benchmark-dir ./benchmark \\
        --slug leaf-classification --submission ./my_submission.csv

    # Batch grade all submissions in a results directory
    python grade_submission.py --benchmark-dir ./benchmark \\
        --results-dir ./benchmark/results/opencode_gpt-4o_full
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_registry(data_dir: Path):
    from mlebench.registry import registry as _reg
    return _reg.set_data_dir(data_dir)


def grade_one(sub_path: Path, slug: str, registry) -> dict:
    """Grade a submission CSV. Returns dict with score, valid_submission, above_median, any_medal."""
    from fetch_leaderboards import ensure_leaderboard
    from mlebench.grade import grade_csv

    try:
        ensure_leaderboard(slug)
        comp = registry.get_competition(slug)
        report = grade_csv(sub_path, comp)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    out = {}
    for k, v in report.__dict__.items():
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark-dir", type=Path, required=True,
                   help="Benchmark directory with data/")
    p.add_argument("--slug", default=None, help="Competition slug (for single grading)")
    p.add_argument("--submission", type=Path, default=None,
                   help="Path to submission CSV (for single grading)")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Results directory to batch-grade (each subdir = slug with _submission.csv)")
    args = p.parse_args()

    data_dir = args.benchmark_dir.resolve() / "data"
    registry = get_registry(data_dir)

    if args.slug and args.submission:
        # Single grading
        report = grade_one(args.submission, args.slug, registry)
        print(json.dumps(report, indent=2, default=str))

    elif args.results_dir:
        # Batch grading
        results_dir = args.results_dir.resolve()
        slugs = sorted(
            d.name for d in results_dir.iterdir()
            if d.is_dir() and (d / "_submission.csv").exists()
        )
        print(f"Grading {len(slugs)} submissions in {results_dir}\n")

        summary = []
        for slug in slugs:
            sub = results_dir / slug / "_submission.csv"
            report = grade_one(sub, slug, registry)
            grade_file = results_dir / slug / "_grade.json"
            grade_file.write_text(json.dumps(report, indent=2, default=str))

            score = report.get("score")
            valid = report.get("valid_submission")
            medal = report.get("any_medal")
            print(f"  {slug}: score={score}, valid={valid}, medal={medal}")
            summary.append({"slug": slug, **report})

        scores = [s["score"] for s in summary if isinstance(s.get("score"), (int, float))]
        valid = sum(1 for s in summary if s.get("valid_submission"))
        medals = sum(1 for s in summary if s.get("any_medal"))
        print(f"\nSummary: {len(slugs)} tasks, {valid} valid, {medals} medals")
        if scores:
            print(f"  Mean score: {sum(scores)/len(scores):.4f}")

    else:
        p.error("Provide either --slug + --submission, or --results-dir")


if __name__ == "__main__":
    main()
