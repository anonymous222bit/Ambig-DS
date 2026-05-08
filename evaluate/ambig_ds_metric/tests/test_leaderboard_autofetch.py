"""Tests for Issue: auto-fetch LFS-stub leaderboard CSVs before grading.

When mle-bench is installed with GIT_LFS_SKIP_SMUDGE=1, leaderboard.csv
files are ~130-byte pointer stubs.  Grading then fails with:

    AssertionError: Leaderboard must have a 'score' column.

The fix adds `ensure_leaderboard(slug)` to fetch_leaderboards.py and
calls it automatically from grade functions before `grade_csv()`.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))


# ── LFS stub fixture ─────────────────────────────────────────────────────────

LFS_POINTER = textwrap.dedent("""\
    version https://git-lfs.github.com/spec/v1
    oid sha256:d5f3dc57f971ea476f1e43ea68ec5f184f0ecb41f56528335d0a1add5eeb19a9
    size 84693
""")

REAL_CSV_HEADER = (
    "TeamName,score\n" + "\n".join(f"team{i},{0.01*i}" for i in range(100))
)


# ═════════════════════════════════════════════════════════════════════════════
# fetch_leaderboards.ensure_leaderboard
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsureLeaderboardExists:
    """ensure_leaderboard() must be importable from fetch_leaderboards."""

    def test_function_exists(self):
        import fetch_leaderboards as fl
        assert callable(getattr(fl, "ensure_leaderboard", None)), \
            "fetch_leaderboards must expose ensure_leaderboard()"


class TestEnsureLeaderboardDetectsStub:
    """ensure_leaderboard() should detect LFS stubs and call fetch_one."""

    def test_stub_triggers_fetch(self, tmp_path):
        """A <500-byte file should be detected as a stub and re-fetched."""
        import fetch_leaderboards as fl

        slug = "fake-competition"
        comp_dir = tmp_path / slug
        comp_dir.mkdir()
        lb = comp_dir / "leaderboard.csv"
        lb.write_text(LFS_POINTER)

        assert lb.stat().st_size < 500, "sanity: LFS pointer is small"

        with patch.object(fl, "COMPS_DIR", tmp_path), \
             patch.object(fl, "fetch_one", return_value=(True, "84,693 bytes")) as mock_fetch:
            fl.ensure_leaderboard(slug)
            mock_fetch.assert_called_once_with(slug)

    def test_real_csv_skips_fetch(self, tmp_path):
        """A valid (>= 500 byte) CSV should NOT trigger a fetch."""
        import fetch_leaderboards as fl

        slug = "real-competition"
        comp_dir = tmp_path / slug
        comp_dir.mkdir()
        lb = comp_dir / "leaderboard.csv"
        lb.write_text(REAL_CSV_HEADER)

        assert lb.stat().st_size >= 500, "sanity: real CSV is large enough"

        with patch.object(fl, "COMPS_DIR", tmp_path), \
             patch.object(fl, "fetch_one") as mock_fetch:
            fl.ensure_leaderboard(slug)
            mock_fetch.assert_not_called()

    def test_missing_file_triggers_fetch(self, tmp_path):
        """A completely missing leaderboard.csv should trigger a fetch."""
        import fetch_leaderboards as fl

        slug = "missing-competition"
        comp_dir = tmp_path / slug
        comp_dir.mkdir()
        # no leaderboard.csv at all

        with patch.object(fl, "COMPS_DIR", tmp_path), \
             patch.object(fl, "fetch_one", return_value=(True, "ok")) as mock_fetch:
            fl.ensure_leaderboard(slug)
            mock_fetch.assert_called_once_with(slug)

    def test_fetch_failure_raises(self, tmp_path):
        """If fetch_one fails, ensure_leaderboard should raise RuntimeError."""
        import fetch_leaderboards as fl

        slug = "broken-competition"
        comp_dir = tmp_path / slug
        comp_dir.mkdir()
        (comp_dir / "leaderboard.csv").write_text(LFS_POINTER)

        with patch.object(fl, "COMPS_DIR", tmp_path), \
             patch.object(fl, "fetch_one", return_value=(False, "404 not found")):
            with pytest.raises(RuntimeError, match="Cannot fetch leaderboard"):
                fl.ensure_leaderboard(slug)


# ═════════════════════════════════════════════════════════════════════════════
# grade functions call ensure_leaderboard before grade_csv
# ═════════════════════════════════════════════════════════════════════════════

class TestGradeCallsEnsureLeaderboard:
    """grade() in step_2_run_agent.py must call ensure_leaderboard."""

    def test_step2_grade_calls_ensure(self):
        import step_2_run_agent as s2

        mock_registry = MagicMock()
        mock_comp = MagicMock()
        mock_registry.get_competition.return_value = mock_comp

        mock_report = MagicMock()
        mock_report.__dict__ = {"score": 0.5, "valid_submission": True}

        with patch("fetch_leaderboards.ensure_leaderboard") as mock_ensure, \
             patch("mlebench.grade.grade_csv", return_value=mock_report):
            result = s2.grade(Path("/fake/sub.csv"), "some-slug", mock_registry)
            mock_ensure.assert_called_once_with("some-slug")

    def test_grade_submission_calls_ensure(self):
        import grade_submission as gs

        mock_registry = MagicMock()
        mock_comp = MagicMock()
        mock_registry.get_competition.return_value = mock_comp

        mock_report = MagicMock()
        mock_report.__dict__ = {"score": 0.5, "valid_submission": True}

        with patch("fetch_leaderboards.ensure_leaderboard") as mock_ensure, \
             patch("mlebench.grade.grade_csv", return_value=mock_report):
            result = gs.grade_one(Path("/fake/sub.csv"), "some-slug", mock_registry)
            mock_ensure.assert_called_once_with("some-slug")


class TestGradeEnsureFailureIsGraceful:
    """If ensure_leaderboard raises, grade should return an error dict (not crash)."""

    def test_step2_grade_returns_error_on_fetch_failure(self):
        import step_2_run_agent as s2

        mock_registry = MagicMock()

        with patch("fetch_leaderboards.ensure_leaderboard",
                    side_effect=RuntimeError("Cannot fetch leaderboard for x: 404")):
            result = s2.grade(Path("/fake/sub.csv"), "x", mock_registry)
            assert "error" in result
            assert "Cannot fetch leaderboard" in result["error"]

    def test_grade_submission_returns_error_on_fetch_failure(self):
        import grade_submission as gs

        mock_registry = MagicMock()

        with patch("fetch_leaderboards.ensure_leaderboard",
                    side_effect=RuntimeError("Cannot fetch leaderboard for x: 404")):
            result = gs.grade_one(Path("/fake/sub.csv"), "x", mock_registry)
            assert "error" in result
            assert "Cannot fetch leaderboard" in result["error"]
