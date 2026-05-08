#!/usr/bin/env python3
"""Step 1 — set up the Ambig-DS-T benchmark from a release tree + upstream DSBench.

Two equivalent sources for the release tree:
  --release-source hf     download prompts + manifests + eval.py from HF
                          (default repo: anonymous222bit/Ambig-DS-T)
  --release-source local  copy a release/ tree built locally by
                          create_datasets/ambig_ds_target/pipeline_DSBench/
                          step_4_build_release.py (use --release-path).

Steps:
  1. Install <bench>/release/ from the chosen source.
  2. Locate upstream DSBench `data_resplit/<slug>/` CSVs (passed via
     --dsbench-data-root or $DSBENCH_DATA_ROOT) and copy them into
     <bench>/data/<slug>/full/.
  3. Re-derive the target-ambiguous CSVs from each per-task manifest by
     calling `process_task()` from `create_datasets/ambig_ds_target/
     pipeline_DSBench/step_1_generate_decoy.py`. Outputs land in
     <bench>/data/<slug>/ambig/.
  4. Verify that every task has prompts + full data + ambig data.

Filesystem layout produced
--------------------------
<bench>/
├── release/                          # mirror of HF repo contents
│   ├── README.md
│   ├── tasks.csv
│   └── tasks/<slug>/
│       ├── task.txt                  # full / non-ambiguous prompt
│       ├── task_ambig.txt            # target-ambiguous prompt
│       ├── eval.py                   # DSBench-style evaluator
│       └── _manifest.json            # decoy recipe + diagnostics
├── data/<slug>/
│   ├── full/                         # original DSBench CSVs
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── test_answer.csv
│   │   └── sample_submission.csv
│   └── ambig/                        # rebuilt decoy CSVs (no targets in test)
│       ├── train.csv                 # has [id, f_*, val_1, val_2]
│       ├── test.csv                  # has [id, f_*]
│       └── _manifest.json            # decoy recipe (copy)
└── task_list.txt                     # one slug per line

Usage
-----
    # Path A: from HF
    python step_1_setup_benchmark.py \
        --benchmark-dir ./benchmark \
        --dsbench-data-root /path/to/Dataset/data_modeling/data/data

    # Path B: from a local pipeline build
    python step_1_setup_benchmark.py \
        --benchmark-dir ./benchmark \
        --release-source local \
        --release-path /path/to/pipeline_DSBench/release \
        --dsbench-data-root /path/to/Dataset/data_modeling/data/data

    # Just verify
    python step_1_setup_benchmark.py --benchmark-dir ./benchmark --verify-only

    # Subset
    python step_1_setup_benchmark.py --benchmark-dir ./benchmark \
        --dsbench-data-root /path/to/dsbench/data \
        --tasks playground-series-s3e17,playground-series-s3e19
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # _gh_work/DSBench
PIPELINE_DIR = REPO_ROOT / "create_datasets" / "ambig_ds_target" / "pipeline_DSBench"


def _import_decoy_module():
    """Import process_task() from pipeline_DSBench/step_1_generate_decoy.py."""
    if not (PIPELINE_DIR / "step_1_generate_decoy.py").exists():
        sys.exit(
            f"missing decoy generator at {PIPELINE_DIR / 'step_1_generate_decoy.py'}\n"
            "  this script reuses pipeline_DSBench/step_1_generate_decoy.py to\n"
            "  rebuild ambig CSVs from manifests; the file must be in the same repo."
        )
    sys.path.insert(0, str(PIPELINE_DIR))
    import step_1_generate_decoy  # noqa: F401
    return step_1_generate_decoy


# --------------------------------------------------------------------------- #
def install_release_from_local(benchmark_dir: Path, src: Path) -> Path:
    """Mirror a locally-built release/ tree (from
    create_datasets/.../pipeline_DSBench/step_4_build_release.py) into
    <bench>/release/. Copies, doesn't symlink, so downstream writes are safe.
    """
    release = benchmark_dir / "release"
    src = Path(src).resolve()
    if not (src / "tasks").exists():
        sys.exit(f"--release-path {src} has no tasks/ subdir; "
                 "run pipeline_DSBench/step_4_build_release.py first")
    if release.exists():
        shutil.rmtree(release)
    print(f"[1/5] Installing local release from {src} ...")
    shutil.copytree(src, release)
    print(f"  -> {release}")
    return release


# --------------------------------------------------------------------------- #
def download_release(benchmark_dir: Path, repo_id: str) -> Path:
    """Download the HF release into <bench>/release/."""
    from huggingface_hub import snapshot_download

    release = benchmark_dir / "release"
    if release.exists() and (release / "tasks").exists() \
            and any((release / "tasks").iterdir()):
        print(f"[1/5] release/ already populated, skipping HF download "
              f"(delete {release} to force).")
        return release

    print(f"[1/5] Downloading HF release ({repo_id}) ...")
    cache_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    release.mkdir(parents=True, exist_ok=True)
    for item in cache_dir.iterdir():
        dst = release / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    print(f"  -> {release}")
    return release


def write_task_list(benchmark_dir: Path, release: Path) -> list[str]:
    tasks_dir = release / "tasks"
    slugs = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
    (benchmark_dir / "task_list.txt").write_text("\n".join(slugs) + "\n")
    return slugs


# --------------------------------------------------------------------------- #
def copy_baselines(slug: str, perf_root: Path, benchmark_dir: Path) -> bool:
    """Copy upstream DSBench save_performance/{GT,baseline}/<slug>/result.txt
    into <bench>/baselines/<slug>/{gt,baseline}.txt for RPG normalization.

    The Relative Performance Gap (RPG) used by the paper is
        max((p - b) / (g - b), 0)
    where g = GT/result.txt and b = baseline/result.txt for that task.
    """
    gt_src = perf_root / "GT" / slug / "result.txt"
    bl_src = perf_root / "baseline" / slug / "result.txt"
    if not gt_src.exists() or not bl_src.exists():
        print(f"  [{slug}] BASELINES: missing (gt={gt_src.exists()}, baseline={bl_src.exists()})")
        return False
    dst = benchmark_dir / "baselines" / slug
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gt_src, dst / "gt.txt")
    shutil.copy2(bl_src, dst / "baseline.txt")
    return True


# --------------------------------------------------------------------------- #
def copy_full_csvs(slug: str, dsbench_root: Path, benchmark_dir: Path) -> bool:
    """Copy upstream DSBench data_resplit/<slug>/* into <bench>/data/<slug>/full/.

    DSBench keeps train/test/sample_submission under `data_resplit/<slug>/`,
    but the held-out `test_answer.csv` lives in the parallel
    `answers/<slug>/test_answer.csv` directory. We pull both.
    """
    src = dsbench_root / "data_resplit" / slug
    if not src.exists():
        print(f"  [{slug}] FULL: source missing ({src})")
        return False
    dst = benchmark_dir / "data" / slug / "full"
    dst.mkdir(parents=True, exist_ok=True)

    # train/test/sample_submission live next to each other under data_resplit/.
    for n in ["train.csv", "test.csv", "sample_submission.csv"]:
        if not (src / n).exists():
            print(f"  [{slug}] FULL: missing {n} in {src}")
            return False
        if not (dst / n).exists():
            shutil.copy2(src / n, dst / n)

    # test_answer.csv may live alongside the data OR in a sibling answers/ dir.
    answer_candidates = [
        src / "test_answer.csv",
        dsbench_root / "answers" / slug / "test_answer.csv",
    ]
    answer_src = next((p for p in answer_candidates if p.exists()), None)
    if answer_src is None:
        print(f"  [{slug}] FULL: test_answer.csv not found "
              f"(checked {[str(p) for p in answer_candidates]})")
        return False
    if not (dst / "test_answer.csv").exists():
        shutil.copy2(answer_src, dst / "test_answer.csv")
    return True


# --------------------------------------------------------------------------- #
def rebuild_ambig_csvs(slug: str, release: Path, dsbench_root: Path,
                       benchmark_dir: Path, decoy_mod) -> bool:
    """Re-derive the ambig CSVs deterministically from the manifest recipe.

    We synthesise a one-row spec CSV plus an argparse Namespace mirroring
    the original pipeline_DSBench/step_1_generate_decoy.py invocation, then
    call its process_task() so the implementation lives in exactly one place.
    """
    manifest_p = release / "tasks" / slug / "_manifest.json"
    if not manifest_p.exists():
        print(f"  [{slug}] AMBIG: manifest missing in release ({manifest_p})")
        return False
    manifest = json.loads(manifest_p.read_text())

    # The HF release uses the PUBLIC restructured schema (task / ambig_recipe /
    # diagnostics). The legacy raw schema (flat keys at top level) is also
    # supported as a fallback for direct re-use of pipeline_DSBench output.
    if "task" in manifest and "ambig_recipe" in manifest:
        task_blk = manifest["task"]
        recipe_blk = manifest["ambig_recipe"]
        original_target = task_blk["original_target_name"]
        target_type = task_blk["task_type"]
        n_train = task_blk["n_train"]
        n_test = task_blk["n_test"]
        n_features = task_blk["n_features"]
        noise_cls = recipe_blk["noise_classification_frac"]
        noise_reg = recipe_blk["noise_regression_sigma_frac"]
        seed_master = recipe_blk["seeds"]["master"]
        decoy_method = recipe_blk["method"]
    else:
        original_target = manifest["original_target_name"]
        target_type = manifest["target_type"]
        n_train = manifest["n_train"]
        n_test = manifest["n_test"]
        n_features = manifest["n_features"]
        noise_cls = manifest["noise_classification_frac"]
        noise_reg = manifest["noise_regression_sigma_frac"]
        seed_master = manifest["seeds"]["master"]
        decoy_method = manifest.get("decoy_method", "")

    # Where the ambig CSVs will end up (process_task writes <out_root>/data/<slug>/).
    ambig_out = benchmark_dir / "data" / slug / "ambig"
    train_dst = ambig_out / "train.csv"
    test_dst = ambig_out / "test.csv"
    if train_dst.exists() and test_dst.exists():
        return True

    # Synthesise the per-task row that process_task expects.
    row = pd.Series({
        "task": slug,
        "target_name": original_target,
        "target_type": target_type,
        "n_unique": 0,  # only used for logging
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "modality": "tabular",
        "group": "hf_release",
        "notes": "",
    })

    # process_task reads from <args.src_data_root>/<task>/{train,test}.csv
    # and writes to <args.out_root>/data/<task>/...  (note the extra "data/").
    # We point at a per-call tempdir, then move the produced files to
    # <bench>/data/<slug>/ambig/ so the layout the rest of the evaluator
    # expects is preserved.
    cal = manifest.get("calibration", {})
    tmp_out = tempfile.mkdtemp(prefix=f"ambig_rebuild_{slug}_")
    args = Namespace(
        src_data_root=str(dsbench_root / "data_resplit"),
        out_root=tmp_out,
        seed=seed_master,
        noise_classification=noise_cls,
        noise_regression=noise_reg,
        cv_tolerance=cal.get("cv_tolerance", 0.02),
        bisection_steps=cal.get("bisection_steps", 12),
        max_noise=cal.get("max_noise", 0.80),
        apply_dtype_snap=("dtype_match" in decoy_method),
        pool_min=3,
        pool_max=8,
        pool_frac=0.6,
        # Paper Sec 4.2 correlation filter (see step_1_generate_decoy.py).
        # Disabled here (cap=1.0) because re-derivation must reproduce the
        # exact decoy that the manifest's recorded seed produced; resampling
        # would diverge. Filter belongs to BUILD time, not REBUILD time.
        max_abs_correlation=1.0,
        corr_filter_seeds=0,
        max_cv_rows=20000,
        # Re-derivation MUST use the same train-row count the manifest was
        # built with, otherwise the recorded seed produces a different
        # decoy permutation. n_train comes from the manifest's task block.
        max_train_rows=int(n_train),
        unlearnable_truth_threshold=cal.get(
            "unlearnable_truth_threshold", 0.10),
    )
    master_rng = np.random.default_rng(args.seed)

    try:
        decoy_mod.process_task(row, args, master_rng)
    except Exception as e:
        print(f"  [{slug}] AMBIG: rebuild FAILED: {e}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False

    # process_task writes to <out_root>/data/<task>/.
    written = Path(tmp_out) / "data" / slug
    train_src = written / "train.csv"
    test_src = written / "test.csv"
    sub_src = written / "sample_submission.csv"
    man_src = written / "_manifest.json"
    if not train_src.exists():
        print(f"  [{slug}] AMBIG: process_task did not produce train.csv at {train_src}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False
    ambig_out.mkdir(parents=True, exist_ok=True)
    shutil.move(str(train_src), train_dst)
    shutil.move(str(test_src), test_dst)
    if sub_src.exists():
        shutil.move(str(sub_src), ambig_out / "sample_submission.csv")
    if man_src.exists():
        shutil.move(str(man_src), ambig_out / "_manifest.json")
    shutil.rmtree(tmp_out, ignore_errors=True)
    return True


# --------------------------------------------------------------------------- #
def verify(benchmark_dir: Path) -> bool:
    print("[5/5] Verifying ...")
    task_list = benchmark_dir / "task_list.txt"
    if not task_list.exists():
        print("  ERROR: task_list.txt missing")
        return False
    slugs = [s.strip() for s in task_list.read_text().splitlines() if s.strip()]
    release = benchmark_dir / "release"

    n_release = n_full = n_ambig = n_baseline = 0
    issues: list[str] = []
    for slug in slugs:
        rel = release / "tasks" / slug
        rel_ok = all((rel / n).exists() for n in
                     ["task.txt", "task_ambig.txt", "eval.py", "_manifest.json"])
        if rel_ok:
            n_release += 1
        else:
            issues.append(f"  release incomplete: {slug}")

        full = benchmark_dir / "data" / slug / "full"
        full_ok = all((full / n).exists() for n in
                      ["train.csv", "test.csv", "test_answer.csv",
                       "sample_submission.csv"])
        if full_ok:
            n_full += 1
        else:
            issues.append(f"  full data missing/incomplete: {slug}")

        amb = benchmark_dir / "data" / slug / "ambig"
        amb_ok = (amb / "train.csv").exists() and (amb / "test.csv").exists()
        if amb_ok:
            n_ambig += 1
        else:
            issues.append(f"  ambig data missing/incomplete: {slug}")

        bl = benchmark_dir / "baselines" / slug
        bl_ok = (bl / "gt.txt").exists() and (bl / "baseline.txt").exists()
        if bl_ok:
            n_baseline += 1
        else:
            issues.append(f"  baselines missing (gt/baseline.txt): {slug}")

    print(f"  Tasks:        {len(slugs)}")
    print(f"  Release ok:   {n_release}/{len(slugs)}")
    print(f"  Full data:    {n_full}/{len(slugs)}")
    print(f"  Ambig data:   {n_ambig}/{len(slugs)}")
    print(f"  Baselines:    {n_baseline}/{len(slugs)}")
    for iss in issues[:20]:
        print(iss)
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more")
    return n_release == n_full == n_ambig == n_baseline == len(slugs)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True)
    ap.add_argument("--release-source", choices=["hf", "local"], default="hf",
                    help="Where to obtain the release/ tree from (default: hf)")
    ap.add_argument("--hf-repo", default="anonymous222bit/Ambig-DS-T",
                    help="HF dataset id when --release-source=hf")
    ap.add_argument("--release-path", type=Path, default=None,
                    help="Local release/ dir (from pipeline_DSBench/"
                         "step_4_build_release.py) when --release-source=local")
    ap.add_argument("--dsbench-data-root",
                    default=os.environ.get("DSBENCH_DATA_ROOT", ""),
                    help="Path to upstream DSBench's `Dataset/data_modeling/"
                         "data/data` (must contain `data_resplit/<slug>/...`). "
                         "Falls back to $DSBENCH_DATA_ROOT.")
    ap.add_argument("--dsbench-perf-root",
                    default=os.environ.get("DSBENCH_PERF_ROOT", ""),
                    help="Path to upstream DSBench's unzipped `save_performance/` "
                         "directory (must contain `GT/<slug>/result.txt` and "
                         "`baseline/<slug>/result.txt`). Falls back to "
                         "$DSBENCH_PERF_ROOT, then to <dsbench-data-root>/../save_performance.")
    ap.add_argument("--tasks", default="",
                    help="Comma-separated subset of slugs to set up (default: all).")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    bench = args.benchmark_dir.resolve()
    bench.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        verify(bench)
        return

    if args.release_source == "local":
        if args.release_path is None:
            sys.exit("--release-source=local requires --release-path PATH")
        release = install_release_from_local(bench, args.release_path)
    else:
        release = download_release(bench, args.hf_repo)
    slugs = write_task_list(bench, release)
    if args.tasks:
        wanted = {s.strip() for s in args.tasks.split(",") if s.strip()}
        slugs = [s for s in slugs if s in wanted]
    if not slugs:
        sys.exit("no slugs to process")

    if not args.dsbench_data_root:
        sys.exit(
            "Need --dsbench-data-root pointing at upstream DSBench's "
            "Dataset/data_modeling/data/data (or set $DSBENCH_DATA_ROOT)."
        )
    dsbench_root = Path(args.dsbench_data_root).resolve()
    if not (dsbench_root / "data_resplit").exists():
        sys.exit(f"--dsbench-data-root {dsbench_root} has no data_resplit/ subdir")

    print(f"[2/5] Copying upstream full CSVs from {dsbench_root}/data_resplit ...")
    n_full_ok = sum(copy_full_csvs(s, dsbench_root, bench) for s in slugs)
    print(f"  ok: {n_full_ok}/{len(slugs)}")

    print(f"[3/5] Rebuilding ambig CSVs from manifests ...")
    decoy_mod = _import_decoy_module()
    n_ambig_ok = sum(
        rebuild_ambig_csvs(s, release, dsbench_root, bench, decoy_mod)
        for s in slugs
    )
    print(f"  ok: {n_ambig_ok}/{len(slugs)}")

    perf_root_arg = args.dsbench_perf_root
    if not perf_root_arg:
        # Default: assume save_performance/ sits next to data_modeling/data/.
        # dsbench_root is .../data_modeling/data, so ../save_performance is
        # .../data_modeling/save_performance.
        cand = (dsbench_root.parent / "save_performance").resolve()
        if cand.exists():
            perf_root_arg = str(cand)
    if not perf_root_arg:
        sys.exit(
            "Need --dsbench-perf-root pointing at upstream DSBench's "
            "save_performance/ (containing GT/<slug>/result.txt and "
            "baseline/<slug>/result.txt). Set $DSBENCH_PERF_ROOT or pass the flag."
        )
    perf_root = Path(perf_root_arg).resolve()
    if not (perf_root / "GT").exists() or not (perf_root / "baseline").exists():
        sys.exit(f"--dsbench-perf-root {perf_root} missing GT/ or baseline/ subdir")
    print(f"[4/5] Copying RPG baselines from {perf_root} ...")
    n_bl_ok = sum(copy_baselines(s, perf_root, bench) for s in slugs)
    print(f"  ok: {n_bl_ok}/{len(slugs)}")

    verify(bench)


if __name__ == "__main__":
    main()
