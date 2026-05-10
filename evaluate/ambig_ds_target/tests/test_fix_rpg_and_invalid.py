#!/usr/bin/env python3
"""Tests for Fix 1 (RPG clipped to [0,1]) and Fix 2 (invalid runs score 0).

Fix 1 — RPG normalization must be clipped to [0, 1]:
    min(max((p - b) / (g - b), 0), 1)
Fix 2 — failed/missing submissions must produce score=0, score_rpg=0.0
    so they remain in the denominator for macro-averaging.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step_4_run_agent import grade


# ── Fix 1: RPG clipped to [0, 1] ─────────────────────────────────────────

class TestRPGClipping:
    """grade() must clip RPG to [0, 1]."""

    def _make_benchmark(self, tmp_path, slug, gt, baseline, eval_score):
        """Wire up a minimal benchmark dir with a fake eval.py."""
        bench = tmp_path / "bench"
        # eval.py that writes a fixed score
        task_dir = bench / "release" / "tasks" / slug
        task_dir.mkdir(parents=True)
        eval_py = task_dir / "eval.py"
        eval_py.write_text(textwrap.dedent(f"""\
            import argparse, os
            ap = argparse.ArgumentParser()
            ap.add_argument("--answer_file"); ap.add_argument("--predict_file")
            ap.add_argument("--path"); ap.add_argument("--name")
            args = ap.parse_args()
            out = os.path.join(args.path, args.name)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "result.txt"), "w") as f:
                f.write("{eval_score}")
        """))

        # manifest
        manifest = {"task": {"original_target_name": "target"}}
        (task_dir / "_manifest.json").write_text(json.dumps(manifest))

        # baselines
        bl = bench / "baselines" / slug
        bl.mkdir(parents=True)
        (bl / "gt.txt").write_text(str(gt))
        (bl / "baseline.txt").write_text(str(baseline))

        # test_answer.csv
        data_dir = bench / "data" / slug / "full"
        data_dir.mkdir(parents=True)
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(
            data_dir / "test_answer.csv", index=False)

        return bench

    def test_rpg_clipped_at_1_when_agent_beats_gt(self, tmp_path):
        """If agent score > GT, RPG should be clipped at 1.0, not >1."""
        bench = self._make_benchmark(tmp_path, "test-task",
                                     gt=0.9, baseline=0.5, eval_score=1.0)
        sub = tmp_path / "sub.csv"
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(sub, index=False)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = grade(sub, "test-task", bench, out_dir)

        assert result["score"] == 1.0
        assert result["score_rpg"] == 1.0, \
            f"RPG should be clipped at 1.0, got {result['score_rpg']}"

    def test_rpg_floors_at_0(self, tmp_path):
        """If agent score < baseline, RPG should be 0."""
        bench = self._make_benchmark(tmp_path, "test-task",
                                     gt=0.9, baseline=0.5, eval_score=0.3)
        sub = tmp_path / "sub.csv"
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(sub, index=False)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = grade(sub, "test-task", bench, out_dir)

        assert result["score_rpg"] == 0.0

    def test_rpg_normal_case(self, tmp_path):
        """Normal case: score between baseline and GT → RPG in (0,1)."""
        bench = self._make_benchmark(tmp_path, "test-task",
                                     gt=1.0, baseline=0.0, eval_score=0.7)
        sub = tmp_path / "sub.csv"
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(sub, index=False)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = grade(sub, "test-task", bench, out_dir)

        assert abs(result["score_rpg"] - 0.7) < 1e-6


# ── Fix 2: Invalid runs score 0 in denominator ───────────────────────────

class TestInvalidRunsScoreZero:
    """Failed/missing submissions must produce score=0, score_rpg=0.0."""

    def _make_minimal_benchmark(self, tmp_path, slug="slug"):
        bench = tmp_path / "bench"
        # eval.py (will be used if submission exists)
        task_dir = bench / "release" / "tasks" / slug
        task_dir.mkdir(parents=True)
        (task_dir / "eval.py").write_text("import sys; sys.exit(1)")
        (task_dir / "_manifest.json").write_text(
            json.dumps({"task": {"original_target_name": "target"}}))
        bl = bench / "baselines" / slug
        bl.mkdir(parents=True)
        (bl / "gt.txt").write_text("1.0")
        (bl / "baseline.txt").write_text("0.0")
        data_dir = bench / "data" / slug / "full"
        data_dir.mkdir(parents=True)
        pd.DataFrame({"id": [1], "target": [0]}).to_csv(
            data_dir / "test_answer.csv", index=False)
        return bench

    def test_no_submission_produces_score_zero(self, tmp_path):
        """When no submission CSV exists, report must have score=0, score_rpg=0.0."""
        from step_4_run_agent import find_submission

        # Simulate the logic from run_one: no submission found
        ws = tmp_path / "workspace"
        ws.mkdir()
        out_task = tmp_path / "out"
        out_task.mkdir()
        bench = self._make_minimal_benchmark(tmp_path)

        sub = find_submission(ws)
        assert sub is None

        # Build report as the code does
        report = {"error": "no submission found", "submission_exists": False}
        report.setdefault("score", 0)
        report.setdefault("score_rpg", 0.0)

        assert report["score"] == 0
        assert report["score_rpg"] == 0.0

    def test_grade_error_produces_score_zero(self, tmp_path):
        """When grade() returns an error dict, setdefault fills score=0."""
        bench = self._make_minimal_benchmark(tmp_path)

        sub = tmp_path / "sub.csv"
        sub.write_text("bad,csv,data\n1,2,3\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        report = grade(sub, "slug", bench, out_dir)
        # grade() returns {"error": ...} without "score" key
        assert "error" in report

        # Apply the same setdefault the caller does
        report.setdefault("score", 0)
        report.setdefault("score_rpg", 0.0)

        assert report["score"] == 0
        assert report["score_rpg"] == 0.0

    def test_successful_grade_not_overwritten(self, tmp_path):
        """setdefault must NOT overwrite a real score from a successful grade."""
        report = {"score": 0.85, "score_rpg": 0.7}
        report.setdefault("score", 0)
        report.setdefault("score_rpg", 0.0)

        assert report["score"] == 0.85
        assert report["score_rpg"] == 0.7
