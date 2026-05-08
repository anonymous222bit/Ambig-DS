#!/usr/bin/env python3
"""Tests for step_3_inferability_audit.py fixes:
  Issue 1 — _find_sample_submission fallback logic
  Issue 2 — main() gracefully handles empty output CSV
"""
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

# Make sure we can import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_3_inferability_audit import _find_sample_submission


# ── Issue 1: _find_sample_submission ──────────────────────────────────────

class TestFindSampleSubmission:
    """Tests for the _find_sample_submission helper."""

    def test_finds_in_ambig_dir(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        ambig.mkdir(parents=True)
        target = ambig / "sample_submission.csv"
        target.write_text("id,target\n1,0\n")
        assert _find_sample_submission(ambig) == target

    def test_finds_sampleSubmission_in_ambig(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        ambig.mkdir(parents=True)
        target = ambig / "sampleSubmission.csv"
        target.write_text("id,target\n1,0\n")
        assert _find_sample_submission(ambig) == target

    def test_prefers_sample_submission_over_sampleSubmission(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        ambig.mkdir(parents=True)
        (ambig / "sample_submission.csv").write_text("id,target\n1,0\n")
        (ambig / "sampleSubmission.csv").write_text("id,target\n1,0\n")
        assert _find_sample_submission(ambig).name == "sample_submission.csv"

    def test_falls_back_to_full_dir(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        full = tmp_path / "slug" / "full"
        ambig.mkdir(parents=True)
        full.mkdir(parents=True)
        target = full / "sample_submission.csv"
        target.write_text("id,target\n1,0\n")
        assert _find_sample_submission(ambig) == target

    def test_falls_back_to_full_sampleSubmission(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        full = tmp_path / "slug" / "full"
        ambig.mkdir(parents=True)
        full.mkdir(parents=True)
        target = full / "sampleSubmission.csv"
        target.write_text("id,target\n1,0\n")
        assert _find_sample_submission(ambig) == target

    def test_raises_when_neither_exists(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        ambig.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No sample submission"):
            _find_sample_submission(ambig)

    def test_raises_when_full_dir_missing_too(self, tmp_path):
        ambig = tmp_path / "slug" / "ambig"
        ambig.mkdir(parents=True)
        # no full/ sibling at all
        with pytest.raises(FileNotFoundError, match="No sample submission"):
            _find_sample_submission(ambig)


# ── Issue 2: empty output CSV guard ───────────────────────────────────────

class TestEmptyOutputGuard:
    """Test that main() does not crash when no audit rows are produced."""

    def test_no_crash_when_all_tasks_fail(self, tmp_path):
        """Simulate: every task fails → out_csv never created → no crash."""
        bench = tmp_path / "bench"
        data = bench / "data"
        audits = bench / "audits" / "inferability"

        # Create one task that will fail (missing train.csv)
        slug_dir = data / "fake-task" / "ambig"
        slug_dir.mkdir(parents=True)
        manifest = {
            "task": "fake-task",
            "true_target_column": "val_1",
            "target_type": "binary",
            "anon_feature_columns": ["f1"],
        }
        (slug_dir / "_manifest.json").write_text(json.dumps(manifest))
        # No train.csv → audit_task will raise → the task is skipped

        sys_argv = [
            "step_3_inferability_audit.py",
            "--benchmark-dir", str(bench),
            "--skip_llm",
            "--skip_strong",
        ]
        with mock.patch("sys.argv", sys_argv):
            # Import main fresh so it picks up the patched argv
            from step_3_inferability_audit import main
            # Should NOT raise; should print the guard message and return
            main()

        # The output CSV should not exist (no rows produced)
        out_csv = audits / "inferability_audit.csv"
        assert not out_csv.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
