"""conftest.py – ensure evaluate/ambig_ds_metric/ is first on sys.path so that
bare ``import agents`` etc. resolve to the metric versions, not the target ones."""
import sys
from pathlib import Path

import pytest

EVAL_DIR = str(Path(__file__).resolve().parent.parent)
_TARGET_DIR = str(Path(__file__).resolve().parent.parent.parent / "ambig_ds_target")

_SHARED_NAMES = ["agents", "clarify_answerer", "step_2_run_agent",
                 "step_3_run_agent_clarify", "_llm_client"]

# ── collection-time setup ──
# Ensure EVAL_DIR is on sys.path so module-level imports in test files work.
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)


@pytest.fixture(autouse=True)
def _metric_sys_path():
    """Per-test fixture: evict same-named modules from evaluate/ambig_ds_target/
    and ensure the metric versions are imported instead."""
    # Evict any target versions cached by earlier tests
    for n in _SHARED_NAMES:
        mod = sys.modules.get(n)
        if mod and hasattr(mod, "__file__") and mod.__file__ and _TARGET_DIR in mod.__file__:
            del sys.modules[n]

    # Temporarily prioritise EVAL_DIR
    old_path = sys.path[:]
    clean = [p for p in sys.path if p != _TARGET_DIR]
    if clean[0:1] != [EVAL_DIR]:
        clean.insert(0, EVAL_DIR)
    sys.path[:] = clean

    yield

    # Restore sys.path and evict our versions so target tests stay clean
    sys.path[:] = old_path
    for n in _SHARED_NAMES:
        mod = sys.modules.get(n)
        if mod and hasattr(mod, "__file__") and mod.__file__ and EVAL_DIR in mod.__file__:
            del sys.modules[n]
