#!/usr/bin/env python3
"""Set up the Ambig-DS-M benchmark from HuggingFace + Kaggle.

Steps:
  1. Download prompts & metadata from HuggingFace (anonymous222bit/Ambig-DS-M)
  2. Restrict task_list.txt to the 67-task evaluation scope.
     Pass --keep-all-82 to disable.
  3. Download competition data from Kaggle via MLE-bench
  4. Verify all tasks are ready

Prerequisites:
  pip install huggingface_hub mlebench kaggle
  # Kaggle API credentials: ~/.kaggle/kaggle.json
  # You must accept competition rules on kaggle.com for each competition

Usage:
    python step_1_setup_benchmark.py --benchmark-dir ./benchmark   # download everything
    python step_1_setup_benchmark.py --benchmark-dir ./benchmark --skip-data  # prompts only
    python step_1_setup_benchmark.py --benchmark-dir ./benchmark --tasks aerial-cactus-identification,dog-breed-identification
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Eval scope: 67-task subset of the 82-task HF dataset.
#
# The HF release ships all 82 competitions; this script (and every downstream
# step that reads task_list.txt) restricts evaluation to the 67 listed below
# by rewriting task_list.txt after the HF download.
#
# Pass --keep-all-82 to disable the filter.
# ──────────────────────────────────────────────────────────────────────────────

EXCLUDED_TASKS = {
    "hubmap-kidney-segmentation",
    "seti-breakthrough-listen",
    "vinbigdata-chest-xray-abnormalities-detection",
    "AI4Code",
    "inaturalist-2019-fgvc6",
    "cdiscount-image-classification-challenge",
    "iwildcam-2019-fgvc6",
    "freesound-audio-tagging-2019",
    "multi-modal-gesture-recognition",
    "spaceship-titanic",
    "dogs-vs-cats-redux-kernels-edition",
    "tabular-playground-series-may-2022",
    "tabular-playground-series-dec-2021",
    "playground-series-s3e18",
    "ml2021spring-hw2",
}


def apply_eval_scope(benchmark_dir: Path, keep_all: bool) -> None:
    """Rewrite task_list.txt to the 67-task evaluation scope.

    Skips quietly if --keep-all-82 was passed or task_list.txt already filtered.
    """
    task_list = benchmark_dir / "task_list.txt"
    if not task_list.exists():
        return
    tasks = [l.strip() for l in task_list.read_text().splitlines() if l.strip()]
    if keep_all:
        print(f"  --keep-all-82: leaving task_list.txt at {len(tasks)} tasks.")
        return
    kept = [t for t in tasks if t not in EXCLUDED_TASKS]
    dropped = [t for t in tasks if t in EXCLUDED_TASKS]
    if not dropped:
        return  # already filtered
    task_list.write_text("\n".join(kept) + "\n")
    print(f"  Eval scope: kept {len(kept)} / {len(tasks)} tasks "
          f"(dropped {len(dropped)}).")
    print(f"  Pass --keep-all-82 to disable this filter.")


def download_hf_dataset(benchmark_dir: Path, repo_id: str = "anonymous222bit/Ambig-DS-M"):
    """Download prompts and metadata from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"\n[1/3] Downloading prompts from HuggingFace ({repo_id})...")
    prompts_dir = benchmark_dir / "prompts"
    if prompts_dir.exists() and len(list(prompts_dir.iterdir())) > 10:
        print(f"  Prompts already present ({len(list(prompts_dir.iterdir()))} items), skipping download.")
        print("  (Delete prompts/ to re-download)")
        return

    cache_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    cache_path = Path(cache_dir)

    # Copy files into benchmark_dir
    for item in cache_path.iterdir():
        dest = benchmark_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"  Downloaded to {benchmark_dir}")
    task_list = benchmark_dir / "task_list.txt"
    if task_list.exists():
        n = len(task_list.read_text().strip().splitlines())
        print(f"  {n} tasks in task_list.txt")


def download_kaggle_data(benchmark_dir: Path, tasks: list[str] | None = None):
    """Download and prepare competition data via MLE-bench."""
    print(f"\n[2/3] Downloading Kaggle competition data via MLE-bench...")
    data_dir = benchmark_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    task_list = benchmark_dir / "task_list.txt"
    all_tasks = task_list.read_text().strip().splitlines()
    if tasks:
        all_tasks = [t for t in all_tasks if t in tasks]

    # Check which tasks already have prepared data
    todo = []
    for slug in all_tasks:
        public = data_dir / slug / "prepared" / "public"
        if public.exists() and any(public.iterdir()):
            continue
        todo.append(slug)

    if not todo:
        print(f"  All {len(all_tasks)} tasks already have prepared data.")
        return

    print(f"  {len(todo)} tasks need data (out of {len(all_tasks)} total).")
    print(f"  This requires a Kaggle API key (~/.kaggle/kaggle.json)")
    print(f"  and acceptance of each competition's rules on kaggle.com.\n")

    for slug in todo:
        print(f"  Preparing {slug}...")
        cmd = [
            "mlebench", "prepare",
            "-c", slug,
            "--data-dir", str(data_dir),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                print(f"    FAILED: {result.stderr[-300:]}")
            else:
                print(f"    OK")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (>1h)")
        except Exception as e:
            print(f"    ERROR: {e}")


def verify(benchmark_dir: Path) -> bool:
    """Verify benchmark is complete."""
    print(f"\n[3/3] Verifying benchmark...")
    task_list = benchmark_dir / "task_list.txt"
    if not task_list.exists():
        print("  ERROR: task_list.txt missing")
        return False

    tasks = task_list.read_text().strip().splitlines()
    prompts_ok = 0
    data_ok = 0
    issues = []

    for slug in tasks:
        full = benchmark_dir / "prompts" / slug / "full.md"
        ambig = benchmark_dir / "prompts" / slug / "ambig_metric.md"
        public = benchmark_dir / "data" / slug / "prepared" / "public"
        private = benchmark_dir / "data" / slug / "prepared" / "private"

        if full.exists() and ambig.exists():
            prompts_ok += 1
        else:
            issues.append(f"  MISSING PROMPT: {slug} (full={full.exists()}, ambig={ambig.exists()})")

        if public.exists() and any(public.iterdir()):
            if private.exists():
                data_ok += 1
            else:
                issues.append(f"  MISSING PRIVATE DATA: {slug}")
        else:
            issues.append(f"  MISSING DATA: {slug}")

    manifest = benchmark_dir / "metric_manifest.json"
    manifest_ok = manifest.exists()

    print(f"  Tasks:     {len(tasks)}")
    print(f"  Prompts:   {prompts_ok}/{len(tasks)}")
    print(f"  Data:      {data_ok}/{len(tasks)}")
    print(f"  Manifest:  {'OK' if manifest_ok else 'MISSING'}")

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for iss in issues[:20]:
            print(iss)
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")

    ready = prompts_ok == len(tasks) and data_ok == len(tasks) and manifest_ok
    if ready:
        print(f"\n  ALL READY: {len(tasks)} tasks fully set up.")
    else:
        print(f"\n  NOT READY: {len(issues)} issues to resolve.")
        if data_ok < len(tasks):
            print("  Hint: Run with --tasks to prepare specific competitions,")
            print("        or accept competition rules at kaggle.com first.")

    return ready


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benchmark-dir", type=Path, required=True,
                    help="Directory to set up the benchmark in")
    ap.add_argument("--hf-repo", default="anonymous222bit/Ambig-DS-M",
                    help="HuggingFace dataset repo ID")
    ap.add_argument("--skip-data", action="store_true",
                    help="Skip Kaggle data download (prompts only)")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated subset of tasks to download data for")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only verify, don't download anything")
    ap.add_argument("--keep-all-82", action="store_true",
                    help="Disable the 67-task eval-scope filter and keep all "
                         "82 tasks from the HF release in task_list.txt.")
    args = ap.parse_args()

    bd = args.benchmark_dir.resolve()
    bd.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        verify(bd)
        return

    download_hf_dataset(bd, args.hf_repo)
    apply_eval_scope(bd, keep_all=args.keep_all_82)

    if not args.skip_data:
        task_subset = args.tasks.split(",") if args.tasks else None
        download_kaggle_data(bd, task_subset)

    verify(bd)


if __name__ == "__main__":
    main()
