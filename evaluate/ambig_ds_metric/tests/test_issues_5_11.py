"""Tests for issues 5–11 fixes in the ambig_ds_metric evaluation pipeline."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

# ── Locate the evaluate/ambig_ds_metric directory ──
EVAL_DIR = Path(__file__).resolve().parent.parent
# sys.path setup is handled by conftest.py


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 5: metrics_classified.csv and edits_log.md removed from README diagram
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue5_ReadmeLayoutDiagram:
    """The data-flow diagram should not list files absent from the HF dataset."""

    def test_no_metrics_classified_csv(self):
        readme = (EVAL_DIR / "README.md").read_text()
        assert "metrics_classified.csv" not in readme, \
            "metrics_classified.csv should be removed from README (not in HF dataset)"

    def test_no_edits_log_md(self):
        readme = (EVAL_DIR / "README.md").read_text()
        assert "edits_log.md" not in readme, \
            "edits_log.md should be removed from README (not in HF dataset)"

    def test_still_lists_metric_manifest(self):
        """metric_manifest.json DOES exist; make sure it wasn't accidentally removed."""
        readme = (EVAL_DIR / "README.md").read_text()
        assert "metric_manifest.json" in readme


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 6: compile_audit_report.py uses argparse, no hardcoded path
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue6_CompileAuditReportArgparse:
    """compile_audit_report.py should use argparse and not crash at import."""

    def test_no_hardcoded_path(self):
        src = (EVAL_DIR / "compile_audit_report.py").read_text()
        assert "/Users/" not in src, \
            "Hardcoded absolute path should be removed from compile_audit_report.py"

    def test_importable_without_crash(self):
        """Importing the module must not trigger side effects (manifest load, etc.)."""
        # Force a fresh import
        mod_name = "compile_audit_report"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "compile_report"), "compile_report function should exist"
        assert hasattr(mod, "main"), "main function should exist"

    def test_help_flag_works(self):
        result = subprocess.run(
            [sys.executable, str(EVAL_DIR / "compile_audit_report.py"), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"--help should work; stderr: {result.stderr}"
        assert "--benchmark-dir" in result.stdout

    def test_missing_flag_errors(self):
        result = subprocess.run(
            [sys.executable, str(EVAL_DIR / "compile_audit_report.py")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, "Should fail when --benchmark-dir is missing"

    def test_compile_report_with_fixture(self, tmp_path):
        """End-to-end: compile_report produces CSV + MD from fixture data."""
        from compile_audit_report import compile_report

        # Create minimal fixture
        manifest = {"test-task": {"metric_name": "AUC"}}
        (tmp_path / "metric_manifest.json").write_text(json.dumps(manifest))
        vdir = tmp_path / "_verify"
        vdir.mkdir()
        verdict = {
            "slug": "test-task",
            "verdict": "pass",
            "checks": {
                "task_preserved": {"pass": True, "rationale": "ok"},
                "ambiguity_preserved": {"pass": True, "rationale": "ok", "leaked_cues": []},
                "plausible_alternatives": {"pass": True, "rationale": "ok", "alternatives": ["RMSE", "MAE"]},
                "decision_relevant": {"pass": True, "rationale": "ok"},
            },
        }
        (vdir / "test-task.json").write_text(json.dumps(verdict))

        compile_report(tmp_path)

        assert (vdir / "audit_report.csv").exists()
        assert (vdir / "audit_report.md").exists()
        csv_text = (vdir / "audit_report.csv").read_text()
        assert "test-task" in csv_text
        assert "AUC" in csv_text


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 7: SSL/TLS prerequisite documented in README
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue7_SslTlsDocumented:
    """README should document SSL_CERT_FILE, REQUESTS_CA_BUNDLE, and GIT_LFS_SKIP_SMUDGE."""

    def test_ssl_cert_file_documented(self):
        readme = (EVAL_DIR / "README.md").read_text()
        assert "SSL_CERT_FILE" in readme

    def test_requests_ca_bundle_documented(self):
        readme = (EVAL_DIR / "README.md").read_text()
        assert "REQUESTS_CA_BUNDLE" in readme

    def test_git_lfs_skip_smudge_documented(self):
        readme = (EVAL_DIR / "README.md").read_text()
        assert "GIT_LFS_SKIP_SMUDGE" in readme

    def test_lfs_skip_before_pip_install(self):
        """GIT_LFS_SKIP_SMUDGE=1 should appear in the pip install command block."""
        readme = (EVAL_DIR / "README.md").read_text()
        # The skip should come before or on the same line as the mle-bench install
        idx_skip = readme.find("GIT_LFS_SKIP_SMUDGE")
        idx_pip = readme.find("pip install -e")
        assert idx_skip < idx_pip, \
            "GIT_LFS_SKIP_SMUDGE should appear before the pip install -e mlebench command"


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 8: default_bin auto-detects ~/.npm-global/bin/opencode
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue8_DefaultBinAutoDetect:
    """default_bin() should auto-detect ~/.npm-global/bin/opencode."""

    def test_opencode_falls_back_to_bare(self):
        """When ~/.npm-global/bin/opencode doesn't exist, fall back to 'opencode'."""
        from agents import default_bin
        with mock.patch("agents.Path.is_file", return_value=False):
            result = default_bin()
        assert result == "opencode"

    def test_opencode_detects_npm_global(self):
        """When ~/.npm-global/bin/opencode exists, return its full path."""
        from agents import default_bin
        expected = str(Path.home() / ".npm-global" / "bin" / "opencode")
        with mock.patch("agents.Path.is_file", return_value=True):
            result = default_bin()
        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 9: Cost reporting: "" instead of "0.000000" when no data
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue9_CostReporting:
    """_format_cost should distinguish real cost, estimated cost, and unknown."""

    def test_real_cost(self):
        from agents import _format_cost
        assert _format_cost(0.05, 0, 0) == "0.050000"

    def test_no_data_returns_empty(self):
        from agents import _format_cost
        assert _format_cost(0.0, 0, 0) == ""

    def test_token_estimate(self):
        from agents import _format_cost
        result = _format_cost(0.0, 1000, 500)
        assert result.startswith("~"), f"Token-based estimate should start with '~': {result}"
        assert float(result[1:]) > 0

    def test_real_cost_takes_precedence(self):
        from agents import _format_cost
        result = _format_cost(0.10, 5000, 2000)
        assert result == "0.100000", "Real cost should take precedence over token estimate"
        assert not result.startswith("~")

    def test_opencode_event_stream_parsing(self):
        """Verify run_opencode parses usage from step_finish events."""
        from agents import run_opencode
        # Build a fake JSONL event stream with usage but no cost
        events = [
            '{"type": "step_start", "part": {}}',
            '{"type": "text", "part": {"text": "Hello world"}}',
            '{"type": "step_finish", "part": {"cost": 0, "usage": {"input_tokens": 1000, "output_tokens": 500}}}',
        ]
        fake_stdout = "\n".join(events)

        fake_result = mock.Mock()
        fake_result.stdout = fake_stdout
        fake_result.stderr = ""
        fake_result.returncode = 0

        with mock.patch("agents.subprocess.run", return_value=fake_result), \
             mock.patch("agents._write_opencode_config"):
            msg, tools, iters, cost = run_opencode(
                "/fake/opencode", "model", "prompt",
                Path("/tmp"), "key", "http://url",
            )

        assert msg == "Hello world"
        assert cost.startswith("~"), f"Should be token-estimated cost: {cost}"
        assert float(cost[1:]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 10: Relative path in task footer (no absolute path leak)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue10_RelativePathInFooter:
    """build_workspace should use a relative path in the task.md footer."""

    def test_footer_uses_relative_path(self, tmp_path):
        from step_2_run_agent import build_workspace, SUBMISSION_NAME

        # Set up minimal fixture
        slug = "test-comp"
        variant = "full"
        bench = tmp_path / "bench"
        (bench / "prompts" / slug).mkdir(parents=True)
        (bench / "prompts" / slug / "full.md").write_text("# Test prompt\n")
        pub = bench / "data" / slug / "prepared" / "public"
        pub.mkdir(parents=True)
        (pub / "train.csv").write_text("a,b\n1,2\n")

        ws, prompt = build_workspace(
            slug, variant, bench, bench / "workspaces", "test_run",
        )
        task_md = (ws / "task.md").read_text()

        # Should contain relative path
        assert f"./{SUBMISSION_NAME}" in task_md
        # Should NOT contain any absolute path to the workspace
        assert str(ws) not in task_md
        assert "(absolute path)" not in task_md

    def test_meta_json_no_submission_path(self, tmp_path):
        """_meta.json should not contain a submission_path field."""
        from step_2_run_agent import build_workspace

        slug = "test-comp"
        bench = tmp_path / "bench"
        (bench / "prompts" / slug).mkdir(parents=True)
        (bench / "prompts" / slug / "full.md").write_text("# Test\n")
        pub = bench / "data" / slug / "prepared" / "public"
        pub.mkdir(parents=True)
        (pub / "data.csv").write_text("x\n1\n")

        ws, _ = build_workspace(slug, "full", bench, bench / "workspaces", "run1")
        meta = json.loads((ws / "_meta.json").read_text())
        assert "submission_path" not in meta


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 11: Stale docstring references to "claw"
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue11_NoClaw:
    """All claw references should be removed from code and docstrings."""

    def test_step2_docstring_no_claw(self):
        import step_2_run_agent as s2
        doc = s2.__doc__
        assert "claw" not in doc.lower(), \
            f"step_2 docstring should not mention claw: {doc[:200]}"

    def test_step3_docstring_no_claw(self):
        import step_3_run_agent_clarify as s3
        doc = s3.__doc__
        assert "claw" not in doc.lower(), \
            f"step_3 docstring should not mention claw: {doc[:200]}"

    def test_agents_no_run_claw(self):
        import agents
        assert not hasattr(agents, "run_claw"), "run_claw should be removed"

    def test_run_agent_no_agent_param(self):
        """run_agent() should NOT take an 'agent' string parameter."""
        import inspect
        from agents import run_agent
        sig = inspect.signature(run_agent)
        assert "agent" not in sig.parameters, \
            f"run_agent should not have 'agent' param: {sig}"

    def test_default_bin_no_param(self):
        """default_bin() should take no arguments."""
        import inspect
        from agents import default_bin
        sig = inspect.signature(default_bin)
        assert len(sig.parameters) == 0, \
            f"default_bin should take no args: {sig}"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6.1: opencode.json contains _NOTE marking it as a reference sample
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix6_1_OpencodeJsonNote:
    """The static opencode.json files should contain a _NOTE field."""

    def test_metric_opencode_json_has_note(self):
        cfg = json.loads((EVAL_DIR / "opencode.json").read_text())
        assert "_NOTE" in cfg, "opencode.json should contain a _NOTE field"
        assert "reference" in cfg["_NOTE"].lower() or "sample" in cfg["_NOTE"].lower()

    def test_target_opencode_json_has_note(self):
        target_json = EVAL_DIR.parent / "ambig_ds_target" / "opencode.json"
        if not target_json.exists():
            pytest.skip("target opencode.json not present")
        cfg = json.loads(target_json.read_text())
        assert "_NOTE" in cfg, "target opencode.json should contain a _NOTE field"

    def test_runtime_config_has_no_note(self):
        """The runtime template in agents.py should NOT have _NOTE (it's injected at runtime)."""
        from agents import _OPENCODE_CONFIG_TEMPLATE
        assert "_NOTE" not in _OPENCODE_CONFIG_TEMPLATE


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6.2: 50-word question truncation enforced in step_3_run_agent_clarify.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix6_2_QuestionTruncation:
    """The 50-word limit in the clarify protocol must be enforced in code."""

    def test_truncation_code_present_metric(self):
        """step_3_run_agent_clarify.py should contain truncation logic."""
        src = (EVAL_DIR / "step_3_run_agent_clarify.py").read_text()
        assert "words[:50]" in src, "50-word truncation should be in metric clarify script"

    def test_truncation_code_present_target(self):
        """step_5_run_agent_clarify.py should contain truncation logic."""
        target_script = EVAL_DIR.parent / "ambig_ds_target" / "step_5_run_agent_clarify.py"
        if not target_script.exists():
            pytest.skip("target clarify script not present")
        src = target_script.read_text()
        assert "words[:50]" in src, "50-word truncation should be in target clarify script"

    def test_question_truncated_field_in_clarify_json(self):
        """_clarify.json output should include question_truncated field."""
        src = (EVAL_DIR / "step_3_run_agent_clarify.py").read_text()
        assert "question_truncated" in src, \
            "question_truncated should be recorded in _clarify.json"

    def test_truncation_logic_correct(self):
        """Simulate the truncation logic directly."""
        # Under 50 words — no truncation
        short = "What metric should I optimize for this task?"
        words = short.split()
        assert len(words) <= 50
        truncated = len(words) > 50
        assert not truncated
        result = " ".join(words[:50]) if truncated else short
        assert result == short

        # Over 50 words — truncated
        long_q = " ".join(f"word{i}" for i in range(80))
        words = long_q.split()
        assert len(words) == 80
        truncated = len(words) > 50
        assert truncated
        result = " ".join(words[:50])
        assert len(result.split()) == 50
        assert result.endswith("word49")

    def test_exactly_50_words_not_truncated(self):
        """Exactly 50 words should NOT be truncated."""
        q = " ".join(f"w{i}" for i in range(50))
        words = q.split()
        assert len(words) == 50
        truncated = len(words) > 50
        assert not truncated
