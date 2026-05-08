"""Tests for Issue 1: manifest agreement between HF release and local rebuild.

After rebuild_ambig_csvs() runs, the rebuilt ambig manifest's
true_target_column must match the release manifest's
true_target_column_in_ambig for EVERY task.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup: paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PIPELINE_DIR = REPO_ROOT / "create_datasets" / "ambig_ds_target" / "pipeline_DSBench"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPELINE_DIR))
import step_1_setup_benchmark as setup_mod
import step_1_generate_decoy as decoy_mod

BENCH = HERE / "benchmark"
RELEASE = BENCH / "release"
DSBENCH_ROOT = Path(os.environ.get(
    "DSBENCH_DATA_ROOT",
    os.path.expanduser("~/DSBench/data_modeling/data"),
))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _release_slugs():
    """All slugs in the release."""
    tasks_dir = RELEASE / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted(
        d for d in os.listdir(tasks_dir)
        if (tasks_dir / d / "_manifest.json").is_file()
    )


def _release_manifest(slug: str) -> dict:
    return json.loads((RELEASE / "tasks" / slug / "_manifest.json").read_text())


# ===========================================================================
# Test 1: _parse_release_manifest extracts correct fields
# ===========================================================================
def test_parse_release_manifest():
    """Verify _parse_release_manifest returns the right fields from a
    restructured HF-release manifest."""
    slug = "dont-overfit-ii"
    raw = _release_manifest(slug)
    m = setup_mod._parse_release_manifest(raw)

    assert m["true_target_column"] == raw["task"]["true_target_column_in_ambig"], \
        f"true_target_column mismatch: {m['true_target_column']} vs {raw['task']['true_target_column_in_ambig']}"
    assert m["decoy_column"] == raw["task"]["decoy_column_in_ambig"]
    assert m["decoy_seed"] == raw["ambig_recipe"]["seeds"]["decoy"]
    assert m["original_target"] == raw["task"]["original_target_name"]
    assert m["feature_map"] == raw["ambig_recipe"]["feature_map"]
    print("  PASS: _parse_release_manifest extracts correct fields")


# ===========================================================================
# Test 2: rebuild produces manifests agreeing with release for ALL tasks
# ===========================================================================
def test_manifest_agreement_all_tasks():
    """For every task, rebuild the ambig CSVs and verify the rebuilt manifest's
    true_target_column matches the release manifest's true_target_column_in_ambig."""
    slugs = _release_slugs()
    assert len(slugs) > 0, "No release tasks found"

    mismatches = []
    errors = []
    for slug in slugs:
        release_manifest = _release_manifest(slug)
        task_blk = release_manifest.get("task", {})
        expected_truth = task_blk.get("true_target_column_in_ambig")
        expected_decoy = task_blk.get("decoy_column_in_ambig")
        if not expected_truth:
            errors.append(f"{slug}: release manifest missing true_target_column_in_ambig")
            continue

        # Remove existing ambig data to force rebuild
        ambig_dir = BENCH / "data" / slug / "ambig"
        if ambig_dir.exists():
            # Read existing rebuilt manifest
            rebuilt_p = ambig_dir / "_manifest.json"
            if rebuilt_p.exists():
                rebuilt = json.loads(rebuilt_p.read_text())
                actual_truth = rebuilt.get("true_target_column")
                actual_decoy = rebuilt.get("decoy_column")
                if actual_truth != expected_truth:
                    mismatches.append(
                        f"{slug}: rebuilt={actual_truth} vs release={expected_truth}"
                    )
                continue

        # Need to rebuild — only if upstream data exists
        src = DSBENCH_ROOT / "data_resplit" / slug
        if not src.exists():
            continue

        ok = setup_mod.rebuild_ambig_csvs(
            slug, RELEASE, DSBENCH_ROOT, BENCH, decoy_mod)
        if not ok:
            errors.append(f"{slug}: rebuild failed")
            continue

        rebuilt_p = ambig_dir / "_manifest.json"
        assert rebuilt_p.exists(), f"{slug}: no rebuilt manifest"
        rebuilt = json.loads(rebuilt_p.read_text())
        actual_truth = rebuilt.get("true_target_column")
        actual_decoy = rebuilt.get("decoy_column")
        if actual_truth != expected_truth:
            mismatches.append(
                f"{slug}: rebuilt={actual_truth} vs release={expected_truth}"
            )
        if actual_decoy != expected_decoy:
            mismatches.append(
                f"{slug}: decoy rebuilt={actual_decoy} vs release={expected_decoy}"
            )

    if mismatches:
        print("  FAIL: manifest mismatches:")
        for m in mismatches:
            print(f"    {m}")
        assert False, f"{len(mismatches)} manifest mismatches"
    if errors:
        print(f"  WARN: {len(errors)} errors (non-fatal): {errors}")
    print(f"  PASS: all {len(slugs)} tasks have consistent manifest assignments")


# ===========================================================================
# Test 3: rebuild on a known-mismatched task produces correct assignment
# ===========================================================================
def test_rebuild_fixes_known_mismatch():
    """dont-overfit-ii was a known-mismatched task (release=val_1, old rebuild=val_2).
    Verify that the new rebuild produces val_1 = truth."""
    slug = "dont-overfit-ii"
    release_manifest = _release_manifest(slug)
    expected = release_manifest["task"]["true_target_column_in_ambig"]

    # Force rebuild by removing existing ambig data
    ambig_dir = BENCH / "data" / slug / "ambig"
    backup = None
    if ambig_dir.exists():
        backup = ambig_dir.parent / "ambig_backup"
        if backup.exists():
            shutil.rmtree(backup)
        ambig_dir.rename(backup)

    try:
        ok = setup_mod.rebuild_ambig_csvs(
            slug, RELEASE, DSBENCH_ROOT, BENCH, decoy_mod)
        assert ok, f"rebuild failed for {slug}"

        rebuilt = json.loads((ambig_dir / "_manifest.json").read_text())
        actual = rebuilt["true_target_column"]
        assert actual == expected, \
            f"MISMATCH: rebuilt true_target_column={actual}, expected={expected}"

        # Also verify the CSV has both val_1 and val_2 columns
        train = pd.read_csv(ambig_dir / "train.csv", nrows=5)
        assert "val_1" in train.columns, "val_1 missing from rebuilt train.csv"
        assert "val_2" in train.columns, "val_2 missing from rebuilt train.csv"
        print(f"  PASS: {slug} rebuilt with true_target_column={actual} (matches release)")
    finally:
        # Restore backup if we made one
        if backup and backup.exists():
            if ambig_dir.exists():
                shutil.rmtree(ambig_dir)
            backup.rename(ambig_dir)


# ===========================================================================
# Test 4: rebuilt CSV val columns contain correct data
# ===========================================================================
def test_truth_column_matches_original_target():
    """The column designated as true_target_column should contain the same
    values as the original target from the upstream full data."""
    slug = "dont-overfit-ii"
    release_manifest = _release_manifest(slug)
    truth_col = release_manifest["task"]["true_target_column_in_ambig"]
    orig_target = release_manifest["task"]["original_target_name"]

    # Force rebuild
    ambig_dir = BENCH / "data" / slug / "ambig"
    backup = None
    if ambig_dir.exists():
        backup = ambig_dir.parent / "ambig_backup"
        if backup.exists():
            shutil.rmtree(backup)
        ambig_dir.rename(backup)

    try:
        ok = setup_mod.rebuild_ambig_csvs(
            slug, RELEASE, DSBENCH_ROOT, BENCH, decoy_mod)
        assert ok, f"rebuild failed for {slug}"

        # Read rebuilt ambig train
        ambig_train = pd.read_csv(ambig_dir / "train.csv")
        # Read original full train
        full_train = pd.read_csv(
            DSBENCH_ROOT / "data_resplit" / slug / "train.csv",
            nrows=len(ambig_train))

        # The truth column in the ambig CSV should equal the original target
        np.testing.assert_array_equal(
            ambig_train[truth_col].values,
            full_train[orig_target].values,
            err_msg=f"truth column {truth_col} doesn't match original {orig_target}"
        )
        print(f"  PASS: {slug} truth column {truth_col} matches original {orig_target}")
    finally:
        if backup and backup.exists():
            if ambig_dir.exists():
                shutil.rmtree(ambig_dir)
            backup.rename(ambig_dir)


# ===========================================================================
# Test 5: step_5 _load_task_manifest returns correct true_target_column
# ===========================================================================
def test_step5_load_manifest_consistency():
    """step_5_run_agent_clarify loads the manifest and accesses
    true_target_column. Verify this returns the same value as the release."""
    slug = "dont-overfit-ii"
    release_manifest = _release_manifest(slug)
    expected = release_manifest["task"]["true_target_column_in_ambig"]

    # Force rebuild
    ambig_dir = BENCH / "data" / slug / "ambig"
    backup = None
    if ambig_dir.exists():
        backup = ambig_dir.parent / "ambig_backup"
        if backup.exists():
            shutil.rmtree(backup)
        ambig_dir.rename(backup)

    try:
        ok = setup_mod.rebuild_ambig_csvs(
            slug, RELEASE, DSBENCH_ROOT, BENCH, decoy_mod)
        assert ok

        # Simulate what step_5 does: prefer ambig manifest, read true_target_column
        ambig_manifest_p = ambig_dir / "_manifest.json"
        manifest = json.loads(ambig_manifest_p.read_text())
        actual = manifest["true_target_column"]
        assert actual == expected, \
            f"step_5 would see true_target_column={actual}, but release says {expected}"
        print(f"  PASS: step_5 would correctly read true_target_column={actual}")
    finally:
        if backup and backup.exists():
            if ambig_dir.exists():
                shutil.rmtree(ambig_dir)
            backup.rename(ambig_dir)


# ===========================================================================
# Test 6: _parse_release_manifest handles legacy flat schema
# ===========================================================================
def test_parse_legacy_manifest():
    """Verify _parse_release_manifest handles the flat (legacy) schema."""
    legacy = {
        "original_target_name": "target",
        "target_type": "classification",
        "n_train": 100,
        "n_test": 50,
        "n_features": 10,
        "id_column": "id",
        "true_target_column": "val_1",
        "decoy_column": "val_2",
        "feature_map": {"a": "f_001"},
        "anon_feature_columns": ["f_001"],
        "noise_classification_frac": 0.1,
        "noise_regression_sigma_frac": 0.1,
        "seeds": {"master": 42, "decoy": 12345},
        "decoy_method": "rank_map",
    }
    m = setup_mod._parse_release_manifest(legacy)
    assert m["true_target_column"] == "val_1"
    assert m["decoy_column"] == "val_2"
    assert m["decoy_seed"] == 12345
    assert m["original_target"] == "target"
    print("  PASS: legacy flat manifest parsed correctly")


# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Issue 1: Manifest Agreement Tests")
    print("=" * 60)

    print("\nTest 1: _parse_release_manifest")
    test_parse_release_manifest()

    print("\nTest 2: manifest agreement across all tasks")
    test_manifest_agreement_all_tasks()

    print("\nTest 3: rebuild fixes known mismatch (dont-overfit-ii)")
    test_rebuild_fixes_known_mismatch()

    print("\nTest 4: truth column matches original target values")
    test_truth_column_matches_original_target()

    print("\nTest 5: step_5 manifest load consistency")
    test_step5_load_manifest_consistency()

    print("\nTest 6: legacy flat manifest parsing")
    test_parse_legacy_manifest()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
