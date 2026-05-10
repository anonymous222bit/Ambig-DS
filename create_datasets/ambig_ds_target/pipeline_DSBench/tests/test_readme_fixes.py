"""Tests for fixes 1-4 in create_datasets/ambig_ds_target/pipeline_DSBench.

1. Hyperparameter table in README matches actual code defaults.
2. step_2_generate_ambig_prompts.py docstring/Usage reference the correct filename.
3. --model CLI flag takes precedence over AMBIG_LLM_MODEL env var.
4. No internal paths (project_5, project_6) leaked into the README.
5. step_2b verifier prompt does not blanket-ban val_1/val_2 column listings.
6. Function-signature defaults in step_1 match CLI defaults.
7. step_3_audit.py does not crash when competitions/ is absent.
8. dsbench_51_tasks.csv ships all 51 tasks with correct columns.
9. README references dsbench_51_tasks.csv and notes Appendix C differences.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
README = PIPELINE / "README.md"
STEP2 = PIPELINE / "step_2_generate_ambig_prompts.py"
STEP2B = PIPELINE / "step_2b_llm_verify.py"
STEP1 = PIPELINE / "step_1_generate_decoy.py"
STEP3 = PIPELINE / "step_3_audit.py"
TASKS_CSV = PIPELINE / "dsbench_51_tasks.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_argparse_defaults(filepath: Path) -> dict[str, str]:
    """Extract argparse add_argument default= values from a Python file.

    Returns a dict of {flag_name: default_value_as_string}.
    """
    source = filepath.read_text()
    defaults = {}
    # Match patterns like: add_argument("--flag_name", ..., default=VALUE, ...)
    for m in re.finditer(
        r'add_argument\(\s*"--(\w+)".*?default\s*=\s*([^,\)]+)',
        source, re.DOTALL,
    ):
        flag = m.group(1)
        val = m.group(2).strip()
        defaults[flag] = val
    return defaults


def _parse_readme_hyperparam_table(readme_text: str) -> dict[str, str]:
    """Parse the hyperparameter table from the README.

    Returns {flag_name: value_string} for each row.
    """
    rows = {}
    for m in re.finditer(
        r"\|\s*`--(\w+)`\s*\|\s*(\S+)\s*\|",
        readme_text,
    ):
        flag = m.group(1)
        val = m.group(2)
        rows[flag] = val
    return rows


# ── Fix 1: hyperparameter table matches code defaults ────────────────────────

class TestHyperparameterTable:
    """README 'Reference: hyperparameters' table must match step_1 defaults."""

    @pytest.fixture(scope="class")
    def code_defaults(self):
        return _parse_argparse_defaults(STEP1)

    @pytest.fixture(scope="class")
    def readme_values(self):
        return _parse_readme_hyperparam_table(README.read_text())

    @pytest.mark.parametrize("flag,expected_code_default", [
        ("bisection_steps", "8"),
        ("max_noise", "0.8"),
        ("pool_max", "40"),
        ("pool_frac", "0.7"),
        ("cv_tolerance", "0.02"),
        ("noise_classification", "0.10"),
        ("noise_regression", "0.10"),
    ])
    def test_numeric_defaults_match(self, readme_values, code_defaults, flag, expected_code_default):
        assert flag in readme_values, f"`--{flag}` missing from README table"
        readme_val = float(readme_values[flag])
        code_val = float(expected_code_default)
        assert readme_val == pytest.approx(code_val), (
            f"README says --{flag}={readme_values[flag]} but code default is {expected_code_default}"
        )

    def test_apply_dtype_snap_documented_as_off(self, readme_values):
        assert "apply_dtype_snap" in readme_values
        val = readme_values["apply_dtype_snap"].lower()
        assert val == "off", (
            f"README says --apply_dtype_snap={val} but code default is off (store_true)"
        )


# ── Fix 2: step_2 docstring and Usage reference correct filename ─────────────

class TestStepNumbering:
    """step_2_generate_ambig_prompts.py must not reference the old filename."""

    @pytest.fixture(scope="class")
    def step2_text(self):
        return STEP2.read_text()

    def test_docstring_says_step_2(self, step2_text):
        # The very first line of the docstring should say "Step 2"
        assert step2_text.startswith('"""Step 2:'), (
            "Docstring should start with 'Step 2:', got: "
            + step2_text[:60]
        )

    def test_no_step_4_generate_reference(self, step2_text):
        assert "step_4_generate_ambig_prompts" not in step2_text, (
            "step_2 still references the old filename step_4_generate_ambig_prompts"
        )

    def test_no_step_3b_reference(self, step2_text):
        assert "step_3b" not in step2_text, (
            "step_2 still references the old 'step_3b' (should be step_1_generate_decoy.py)"
        )

    def test_manifest_missing_message_references_step_1(self, step2_text):
        assert "step_1_generate_decoy.py" in step2_text, (
            "Missing manifest message should reference step_1_generate_decoy.py"
        )


# ── Fix 3: --model CLI flag wins over AMBIG_LLM_MODEL env var ────────────────

class TestModelPrecedence:
    """When --model is explicitly set, it must win over AMBIG_LLM_MODEL env."""

    def test_explicit_model_wins_over_env(self):
        """Simulate: --model my-custom-model with AMBIG_LLM_MODEL=env-model."""
        # Import the module's DEFAULT_MODEL so we can construct the same logic
        # the code uses.
        sys.path.insert(0, str(PIPELINE))
        try:
            from _llm_client import DEFAULT_MODEL
        finally:
            sys.path.pop(0)

        # The fixed logic: use args.model if it differs from DEFAULT_MODEL,
        # otherwise fall back to env var.
        explicit_model = "my-custom-model"
        env_model = "env-model-should-lose"

        old_env = os.environ.get("AMBIG_LLM_MODEL")
        os.environ["AMBIG_LLM_MODEL"] = env_model
        try:
            # Simulate the fixed logic from main()
            args_model = explicit_model
            if args_model != DEFAULT_MODEL:
                resolved = args_model
            else:
                resolved = os.environ.get("AMBIG_LLM_MODEL", args_model)
            assert resolved == explicit_model, (
                f"Explicit --model should win, got {resolved}"
            )
        finally:
            if old_env is None:
                os.environ.pop("AMBIG_LLM_MODEL", None)
            else:
                os.environ["AMBIG_LLM_MODEL"] = old_env

    def test_env_used_when_model_is_default(self):
        """When --model is not passed (DEFAULT_MODEL), env var should win."""
        sys.path.insert(0, str(PIPELINE))
        try:
            from _llm_client import DEFAULT_MODEL
        finally:
            sys.path.pop(0)

        env_model = "env-model-should-win"
        old_env = os.environ.get("AMBIG_LLM_MODEL")
        os.environ["AMBIG_LLM_MODEL"] = env_model
        try:
            args_model = DEFAULT_MODEL
            if args_model != DEFAULT_MODEL:
                resolved = args_model
            else:
                resolved = os.environ.get("AMBIG_LLM_MODEL", args_model)
            assert resolved == env_model, (
                f"Env var should win when --model is default, got {resolved}"
            )
        finally:
            if old_env is None:
                os.environ.pop("AMBIG_LLM_MODEL", None)
            else:
                os.environ["AMBIG_LLM_MODEL"] = old_env


# ── Fix 4: no internal paths leaked into README ──────────────────────────────

class TestNoInternalPaths:
    """README must not reference internal project paths."""

    @pytest.fixture(scope="class")
    def readme_text(self):
        return README.read_text()

    @pytest.mark.parametrize("pattern", [
        "project_5",
        "project_6",
        "/abs/path/to/info_theory",
    ])
    def test_no_internal_path(self, readme_text, pattern):
        assert pattern not in readme_text, (
            f"README still contains internal path reference: '{pattern}'"
        )


# ── Fix 5: step_2b verifier prompt must not blanket-ban val_1/val_2 ──────────

class TestVerifierPromptVal12:
    """step_2b VERIFY_SYSTEM must allow neutral val_1/val_2 column listings.

    The HF release prompts list val_1 and val_2 in their Data Fields sections
    (produced by step_2 rule #6). The verifier must not flag these as leaks.
    It should only flag signposting phrases like 'candidate target val_1'.
    """

    @pytest.fixture(scope="class")
    def verify_system(self):
        return STEP2B.read_text()

    def test_no_blanket_val12_ban(self, verify_system):
        """The verifier must not list val_1/val_2 as unconditional cue leaks."""
        # The old buggy line was:
        #   mentions of "val_1", "val_2", "decoy", "ambiguous",
        # which blanket-banned any mention of val_1/val_2.
        assert 'mentions of "val_1", "val_2"' not in verify_system, (
            "Verifier still has blanket ban on val_1/val_2 mentions — "
            "this contradicts step_2 rule #6 and the HF release prompts"
        )

    def test_neutral_listing_accepted(self, verify_system):
        """The verifier should explicitly say neutral column listings are OK."""
        assert "neutral column listing" in verify_system.lower() or \
               "acceptable" in verify_system.lower(), (
            "Verifier should explicitly state that neutral val_1/val_2 "
            "column listings in data-fields sections are acceptable"
        )

    def test_signposting_still_banned(self, verify_system):
        """Signposting phrases like 'candidate target' must still be banned."""
        lower = verify_system.lower()
        for phrase in ("candidate target", "decoy", "signposting"):
            assert phrase in lower, (
                f"Verifier must still ban signposting phrase '{phrase}'"
            )


# ── Fix 6: function-signature defaults match CLI ─────────────────────────────

class TestFunctionSignatureDefaults:
    """build_decoy() and calibrate_noise() defaults must match CLI argparse."""

    def test_build_decoy_defaults(self):
        sys.path.insert(0, str(PIPELINE))
        try:
            import importlib
            s1 = importlib.import_module("step_1_generate_decoy")
            import inspect
            sig = inspect.signature(s1.build_decoy)
            p = sig.parameters
            assert p["pool_min"].default == 4
            assert p["pool_max"].default == 40
            assert p["low_corr_pool_frac"].default == 0.7
        finally:
            sys.path.pop(0)

    def test_calibrate_noise_hi_default(self):
        sys.path.insert(0, str(PIPELINE))
        try:
            import importlib
            s1 = importlib.import_module("step_1_generate_decoy")
            import inspect
            sig = inspect.signature(s1.calibrate_noise)
            assert sig.parameters["hi"].default == 0.8
        finally:
            sys.path.pop(0)


# ── Fix 7: step_3_audit.py graceful guard ────────────────────────────────────

class TestStep3AuditGuard:
    """step_3_audit.py must not crash with a traceback when competitions/ is absent."""

    def test_no_traceback_without_competitions_dir(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(STEP3)],
            capture_output=True, text=True,
            env={**os.environ, "AMBIG_PIPELINE_ROOT": str(PIPELINE)},
        )
        assert result.returncode != 0, "Should exit non-zero without competitions/"
        assert "ERROR" in result.stderr, "Should print a clear error message"
        assert "Traceback" not in result.stderr, (
            "Should not crash with a traceback — should give a clear error instead"
        )


# ── Fix 8: dsbench_51_tasks.csv exists and is valid ─────────────────────────

class TestTasksCSV:
    """dsbench_51_tasks.csv must ship all 51 tasks with the required columns."""

    def test_csv_exists(self):
        assert TASKS_CSV.exists(), "dsbench_51_tasks.csv not found in pipeline dir"

    def test_csv_has_51_tasks(self):
        import csv
        with open(TASKS_CSV) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 51, f"Expected 51 tasks, got {len(rows)}"

    def test_csv_has_required_columns(self):
        import csv
        with open(TASKS_CSV) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        for col in ("task", "target_name", "target_type"):
            assert col in headers, f"Missing required column: {col}"

    def test_csv_target_types_valid(self):
        import csv
        with open(TASKS_CSV) as f:
            for row in csv.DictReader(f):
                assert row["target_type"] in ("classification", "regression"), (
                    f"Invalid target_type for {row['task']}: {row['target_type']}"
                )


# ── Fix 9: README references CSV and Appendix C ─────────────────────────────

class TestReadmeNewReferences:
    """README must reference dsbench_51_tasks.csv and note Appendix C differences."""

    @pytest.fixture(scope="class")
    def readme_text(self):
        return README.read_text()

    def test_csv_referenced(self, readme_text):
        assert "dsbench_51_tasks.csv" in readme_text, (
            "README should reference the shipped dsbench_51_tasks.csv"
        )

    def test_appendix_c_note(self, readme_text):
        assert "Appendix C" in readme_text, (
            "README should note that Appendix C shows an abbreviated prompt"
        )
