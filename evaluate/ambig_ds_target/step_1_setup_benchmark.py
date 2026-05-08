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
def _parse_release_manifest(manifest: dict) -> dict:
    """Normalise both HF-release (restructured) and legacy (flat) manifest
    schemas into a uniform dict with the fields the rebuild needs."""
    if "task" in manifest and "ambig_recipe" in manifest:
        task_blk = manifest["task"]
        recipe_blk = manifest["ambig_recipe"]
        return {
            "original_target": task_blk["original_target_name"],
            "target_type": task_blk["task_type"],
            "n_train": task_blk["n_train"],
            "n_test": task_blk["n_test"],
            "n_features": task_blk["n_features"],
            "id_column": task_blk.get("id_column", "id"),
            "true_target_column": task_blk["true_target_column_in_ambig"],
            "decoy_column": task_blk["decoy_column_in_ambig"],
            "feature_map": recipe_blk["feature_map"],
            "anon_feature_columns": recipe_blk["anon_feature_columns"],
            "noise_cls": recipe_blk["noise_classification_frac"],
            "noise_reg": recipe_blk["noise_regression_sigma_frac"],
            "decoy_seed": recipe_blk["seeds"]["decoy"],
            "method": recipe_blk["method"],
        }
    # Legacy flat schema (from pipeline_DSBench direct output).
    return {
        "original_target": manifest["original_target_name"],
        "target_type": manifest["target_type"],
        "n_train": manifest["n_train"],
        "n_test": manifest["n_test"],
        "n_features": manifest["n_features"],
        "id_column": manifest.get("id_column", "id"),
        "true_target_column": manifest["true_target_column"],
        "decoy_column": manifest["decoy_column"],
        "feature_map": manifest["feature_map"],
        "anon_feature_columns": manifest["anon_feature_columns"],
        "noise_cls": manifest["noise_classification_frac"],
        "noise_reg": manifest["noise_regression_sigma_frac"],
        "decoy_seed": manifest["seeds"]["decoy"],
        "method": manifest.get("decoy_method", ""),
    }


def rebuild_ambig_csvs(slug: str, release: Path, dsbench_root: Path,
                       benchmark_dir: Path, decoy_mod) -> bool:
    """Deterministically rebuild ambig CSVs using the release manifest.

    Instead of calling process_task() (which derives its own RNG state and
    may produce different val_1/val_2 assignments), this function reads the
    exact decoy seed, feature map, and column assignment recorded in the
    release manifest and calls the low-level decoy helpers directly.  This
    guarantees the rebuilt data matches the HF release exactly.
    """
    manifest_p = release / "tasks" / slug / "_manifest.json"
    if not manifest_p.exists():
        print(f"  [{slug}] AMBIG: manifest missing in release ({manifest_p})")
        return False
    manifest_raw = json.loads(manifest_p.read_text())
    m = _parse_release_manifest(manifest_raw)

    ambig_out = benchmark_dir / "data" / slug / "ambig"
    train_dst = ambig_out / "train.csv"
    test_dst = ambig_out / "test.csv"
    if train_dst.exists() and test_dst.exists():
        return True

    # ---- read upstream full data ----------------------------------------
    src = dsbench_root / "data_resplit" / slug
    if not src.exists():
        print(f"  [{slug}] AMBIG: upstream data missing ({src})")
        return False
    train = pd.read_csv(src / "train.csv", low_memory=False,
                        nrows=m["n_train"] or None)
    test = pd.read_csv(src / "test.csv", low_memory=False)

    original_target = m["original_target"]
    if original_target not in train.columns:
        print(f"  [{slug}] AMBIG: target {original_target!r} not in train columns")
        return False

    feature_map = m["feature_map"]       # {orig_col: anon_col}
    anon_feat_cols = m["anon_feature_columns"]
    feat_cols = [orig for orig, _anon in feature_map.items()]
    idcol = m["id_column"]
    target_type = m["target_type"].strip().lower()
    decoy_seed = m["decoy_seed"]
    truth_col = m["true_target_column"]  # val_1 or val_2
    decoy_col = m["decoy_column"]        # the other one

    # ---- encode train features for decoy construction -------------------
    X_tr_num = decoy_mod.encode_train_for_model(train, feat_cols)
    y_tr = train[original_target].values

    # ---- detect unlearnable-truth case ----------------------------------
    # When cv_true < threshold the original pipeline used a random
    # permutation of the truth instead of the rank-mapped decoy.
    diag = manifest_raw.get("diagnostics", {})
    cv_true = diag.get("cv_true")
    unlearnable = (cv_true is not None and not np.isnan(cv_true)
                   and cv_true < 0.10)

    # ---- build decoy using the exact recorded seed ----------------------
    if unlearnable:
        rng_perm = np.random.default_rng(decoy_seed)
        val_2_raw = np.asarray(y_tr).copy()
        rng_perm.shuffle(val_2_raw)
    else:
        val_2_raw, _pool_idx, _pool_corrs = decoy_mod.build_decoy(
            X_tr_num, y_tr, target_type=target_type, seed=decoy_seed,
            pool_min=3, pool_max=8, low_corr_pool_frac=0.6,
        )

    # ---- apply noise (same method the release was built with) -----------
    fallback_noise = (m["noise_cls"] if target_type == "classification"
                      else m["noise_reg"])
    if "calibrated" in m["method"]:
        # Re-run the same bisection the original build used; deterministic
        # given identical (val_2_raw, X_tr_num, y_tr, decoy_seed).
        calibrated_v2, _cal_level, _cal_cv, _trace = decoy_mod.calibrate_noise(
            val_2_raw, X_tr_num, y_tr, target_type=target_type,
            seed=decoy_seed, cv_true=cv_true,
            cv_tolerance=0.02, max_cv_rows=20_000,
            n_steps=12, lo=0.0, hi=0.50,
        )
        val_2 = calibrated_v2 if calibrated_v2 is not None else (
            decoy_mod.add_label_noise_classification(val_2_raw, fallback_noise, decoy_seed)
            if target_type == "classification"
            else decoy_mod.add_label_noise_regression(val_2_raw, fallback_noise, decoy_seed)
        )
    elif target_type == "classification":
        val_2 = decoy_mod.add_label_noise_classification(
            val_2_raw, fallback_noise, decoy_seed)
    else:
        val_2 = decoy_mod.add_label_noise_regression(
            val_2_raw, fallback_noise, decoy_seed)

    # ---- optional dtype snap (only if the build method included it) -----
    if "dtype_match" in m["method"]:
        val_2 = decoy_mod.snap_to_truth_dtype(val_2, y_tr, target_type)

    # ---- assemble output frames -----------------------------------------
    # If the id column doesn't exist in the upstream data, fabricate it
    # (matching what process_task does for datasets without a natural id).
    if idcol in train.columns:
        train_ids = train[idcol].values
    else:
        train_ids = np.arange(len(train))
    if idcol in test.columns:
        test_ids = test[idcol].values
    else:
        test_ids = np.arange(len(test))

    out_train = pd.DataFrame({idcol: train_ids})
    for orig, anon in feature_map.items():
        out_train[anon] = train[orig].values
    out_train[truth_col] = y_tr
    out_train[decoy_col] = val_2
    out_train = out_train[[idcol] + anon_feat_cols + ["val_1", "val_2"]]

    out_test = pd.DataFrame({idcol: test_ids})
    for orig, anon in feature_map.items():
        out_test[anon] = test[orig].values
    out_test = out_test[[idcol] + anon_feat_cols]

    # ---- write ----------------------------------------------------------
    ambig_out.mkdir(parents=True, exist_ok=True)
    out_train.to_csv(train_dst, index=False)
    out_test.to_csv(test_dst, index=False)

    # Write a flat manifest consistent with what step_5 / step_7 expect.
    local_manifest = {
        "task": slug,
        "true_target_column": truth_col,
        "decoy_column": decoy_col,
        "original_target_name": original_target,
        "target_type": target_type,
        "id_column": idcol,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(feat_cols),
        "feature_map": feature_map,
        "anon_feature_columns": anon_feat_cols,
        "decoy_method": m["method"],
        "noise_classification_frac": m["noise_cls"],
        "noise_regression_sigma_frac": m["noise_reg"],
        "seeds": {"master": manifest_raw.get("ambig_recipe", manifest_raw)
                  .get("seeds", {}).get("master", 0),
                  "decoy": decoy_seed},
    }
    (ambig_out / "_manifest.json").write_text(
        json.dumps(local_manifest, indent=2))
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
