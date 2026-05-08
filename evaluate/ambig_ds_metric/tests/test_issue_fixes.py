"""Tests for issues 1–3 fixes in the ambig_ds_metric evaluation pipeline."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ── Locate the evaluate/ambig_ds_metric directory ──
EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 1: EXCLUDED_TASKS / apply_eval_scope / --keep-all-82 removed
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue1_DeadCodeRemoved:
    """EXCLUDED_TASKS set and apply_eval_scope() should no longer exist."""

    def test_excluded_tasks_removed(self):
        import step_1_setup_benchmark as s1
        assert not hasattr(s1, "EXCLUDED_TASKS"), \
            "EXCLUDED_TASKS should be removed (dead code; none exist in HF dataset)"

    def test_apply_eval_scope_removed(self):
        import step_1_setup_benchmark as s1
        assert not hasattr(s1, "apply_eval_scope"), \
            "apply_eval_scope() should be removed (no-op; filter matched nothing)"

    def test_keep_all_82_flag_removed(self):
        """The --keep-all-82 CLI flag should no longer be accepted."""
        result = subprocess.run(
            [sys.executable, str(EVAL_DIR / "step_1_setup_benchmark.py"),
             "--benchmark-dir", "/tmp/_test_nonexistent", "--keep-all-82"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, "--keep-all-82 should be rejected"
        assert "unrecognized arguments" in result.stderr or "error" in result.stderr.lower()

    def test_docstring_no_67_or_82_references(self):
        import step_1_setup_benchmark as s1
        doc = s1.__doc__ or ""
        assert "67" not in doc, f"Docstring still references '67': {doc[:200]}"
        assert "82" not in doc, f"Docstring still references '82': {doc[:200]}"

    def test_comments_no_67_references(self):
        """Source code comments should not mention '67-task' or '82-task'."""
        src = (EVAL_DIR / "step_1_setup_benchmark.py").read_text()
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                assert "67-task" not in stripped, f"Comment still references '67-task': {stripped}"
                assert "82-task" not in stripped, f"Comment still references '82-task': {stripped}"


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 2: step_2_audit_prompts.py removed from creation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestIssue2_AuditRemovedFromPipeline:
    """step_2_audit_prompts.py was removed; verify no references remain."""

    def test_audit_script_deleted(self):
        audit_path = EVAL_DIR.parent.parent / "create_datasets" / "ambig_ds_metric" / "pipeline" / "step_2_audit_prompts.py"
        assert not audit_path.exists(), "step_2_audit_prompts.py should be deleted"

    def test_no_reference_in_run_pipeline(self):
        sh = (EVAL_DIR / "run_pipeline.sh").read_text()
        for line in sh.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert "step_2_audit_prompts.py" not in stripped, \
                f"step_2_audit_prompts.py is still called (uncommented): {stripped}"


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 3: parse_judge_output robustness
# ═══════════════════════════════════════════════════════════════════════════════

from step_4_judge_audit import parse_judge_output, LABELS  # noqa: E402


class TestIssue3_ParseJudgeOutput:
    """parse_judge_output must handle common LLM output quirks."""

    # ── already-working cases (regression tests) ──

    def test_clean_json(self):
        raw = json.dumps({
            "label": "Intended",
            "confidence": 0.95,
            "evidence_quotes": ["used log loss"],
            "rationale": "Agent trained a proper model.",
        })
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "Intended"
        assert d["confidence"] == 0.95

    def test_code_fenced_clean_json(self):
        inner = json.dumps({
            "label": "WrongObjective",
            "confidence": 0.8,
            "evidence_quotes": [],
            "rationale": "Trained F1 for AUC task.",
        })
        raw = f"```json\n{inner}\n```"
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "WrongObjective"

    def test_preamble_then_json(self):
        raw = 'Here is my analysis:\n{"label": "Abdicated", "confidence": 0.9, "evidence_quotes": [], "rationale": "Copied sample."}'
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "Abdicated"

    def test_invalid_label_rejected(self):
        raw = json.dumps({"label": "NotALabel", "confidence": 0.5})
        d, msg = parse_judge_output(raw)
        assert d is None
        assert "raw=" in msg or "label" in msg.lower()

    def test_no_json_at_all(self):
        d, msg = parse_judge_output("I cannot determine the label.")
        assert d is None
        assert "No JSON" in msg

    # ── THE BUG: unescaped quotes inside evidence_quotes ──

    def test_unescaped_quotes_in_evidence(self):
        """The exact failure mode from the real run: LLM puts unescaped quotes
        inside the evidence_quotes array, breaking json.loads."""
        raw = textwrap.dedent('''\
            ```json
            {
              "label": "Intended",
              "confidence": 0.95,
              "evidence_quotes": [
                "Logistic Regression (efficient for multi-class text classification)",
                "TF-IDF with up to 5,000 features",
                "id,EAP,HPL,MWS" with probability values like "0.073,0.847,0.079"
              ],
              "rationale": "The agent built a legitimate multi-class logistic regression model."
            }
            ```''')
        d, _ = parse_judge_output(raw)
        assert d is not None, "Should recover via regex fallback"
        assert d["label"] == "Intended"

    def test_unescaped_quotes_wrong_objective(self):
        """Same pattern but with WrongObjective label."""
        raw = '{"label": "WrongObjective", "confidence": 0.85, "evidence_quotes": ["optimized "accuracy" instead of AUC"], "rationale": "Wrong target."}'
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "WrongObjective"

    def test_regex_extracts_confidence(self):
        raw = '{"label": "FormBroken", "confidence": 0.72, "evidence_quotes": ["broke "the" form"], "rationale": "Thresholded probabilities."}'
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "FormBroken"
        assert abs(d.get("confidence", 0) - 0.72) < 0.001

    def test_regex_extracts_rationale(self):
        raw = '{"label": "Invalid", "confidence": 0.99, "evidence_quotes": ["no "submission" file"], "rationale": "No usable submission produced."}'
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "Invalid"
        assert "No usable submission" in d.get("rationale", "")

    # ── all valid labels should be accepted ──

    @pytest.mark.parametrize("label", LABELS)
    def test_all_labels_accepted(self, label):
        raw = json.dumps({"label": label, "confidence": 0.5, "evidence_quotes": [], "rationale": "Test."})
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == label

    # ── edge cases ──

    def test_nested_code_fences(self):
        """Double-fenced (```json ``` wrapping ```)."""
        inner = json.dumps({"label": "Other", "confidence": 0.5, "rationale": "Unclear."})
        raw = f"```\n```json\n{inner}\n```\n```"
        d, _ = parse_judge_output(raw)
        assert d is not None
        assert d["label"] == "Other"

    def test_trailing_whitespace(self):
        raw = '  {"label": "Intended", "confidence": 0.9}  \n\n'
        d, _ = parse_judge_output(raw)
        assert d is not None

    def test_empty_string(self):
        d, msg = parse_judge_output("")
        assert d is None

    def test_only_code_fences(self):
        d, msg = parse_judge_output("```json\n```")
        assert d is None
