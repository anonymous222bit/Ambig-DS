#!/usr/bin/env bash
# End-to-end pipeline for the Ambig-DS-M (metric ambiguity) benchmark.
#
# This script demonstrates the full workflow:
#   1. Setup    : download HF prompts + Kaggle competition data
#   2. Generate : (optional) re-generate ambig_metric.md prompts with an LLM
#   3. Audit    : (optional) verify all redacted prompts are clean of metric leaks
#   4. Run      : execute an LLM agent on full + ambig_metric variants
#                  (--skip-existing: won't re-run tasks that already have _grade.json)
#   5. Judge    : (optional) LLM-classify what metric the agent optimized
#
# Usage:
#   ./run_pipeline.sh <benchmark-dir> <model> [tasks]
# Example:
#   ./run_pipeline.sh ./benchmark gpt-4o leaf-classification,dog-breed-identification
#
# Environment:
#   OPENAI_API_KEY   required
#   OPENAI_BASE_URL  optional (defaults to https://api.openai.com/v1)
#   AGENT_BIN        optional path to the opencode binary (default: auto-detect)
#   RUN_CLARIFY      set to 1 to enable Step 3 (clarify pipeline)
#   RUN_JUDGE        set to 1 to enable Step 4 (LLM-judge audit)
#   ANSWERER_MODEL   oracle model for clarify (default: gpt-4o-mini)
#   TIMEOUT          solve-phase timeout in seconds (default: 1800)
#   ASK_TIMEOUT      ask-phase timeout in seconds (default: 120)
#   N_JUDGES         number of independent judge calls (default: 1; paper uses 5)

set -euo pipefail

BENCH_DIR="${1:-./benchmark}"
MODEL="${2:-gpt-4o-mini}"
TASKS="${3:-all}"

AGENT_BIN="${AGENT_BIN:-}"
if [[ -z "$AGENT_BIN" ]]; then
  if [[ -x "$HOME/.npm-global/bin/opencode" ]]; then
    AGENT_BIN="$HOME/.npm-global/bin/opencode"
  else
    AGENT_BIN="opencode"
  fi
fi
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 2
fi

echo "=========================================================="
echo "Ambig-DS-M pipeline"
echo "  benchmark dir : $BENCH_DIR"
echo "  model         : $MODEL"
echo "  tasks         : $TASKS"
echo "  agent         : opencode ($AGENT_BIN)"
echo "  base url      : $BASE_URL"
echo "=========================================================="

# ── Step 1: Setup ─────────────────────────────────────────────
echo
echo "[1/3] Setup: downloading prompts (HF) + competition data (Kaggle)..."
SETUP_ARGS=(--benchmark-dir "$BENCH_DIR")
if [[ "$TASKS" != "all" ]]; then
  SETUP_ARGS+=(--tasks "$TASKS")
fi
"$PY" "$HERE/step_1_setup_benchmark.py" "${SETUP_ARGS[@]}"

# ── Step 2: (optional) Re-generate ambig prompts with an LLM ──
# By default we use the prompts shipped on HF. Uncomment to regenerate.
#
# echo
# echo "[1.5] (optional) Generating ambiguous prompts with LLM..."
# "$PY" "$HERE/../../create_datasets/ambig_ds_metric/pipeline/step_1_generate_ambig_prompts.py" --benchmark-dir "$BENCH_DIR" --run

# ── Step 3: (optional) Audit redacted prompts ─────────────────
# The static auditor (step_2_audit_prompts.py) has been removed from the
# creation pipeline. Step 2 (LLM verify) covers the same ground via the
# paper's four-item checklist. See create_datasets/ambig_ds_metric/README.md.

# ── Step 4: Run agent on both variants ────────────────────────
echo
echo "[2/3] Running agent on FULL variant..."
"$PY" "$HERE/step_2_run_agent.py" \
  --benchmark-dir "$BENCH_DIR" \
  --variant full \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --agent-bin "$AGENT_BIN" \
  --base-url "$BASE_URL" \
  --skip-existing

echo
echo "[3/3] Running agent on AMBIG_METRIC variant..."
"$PY" "$HERE/step_2_run_agent.py" \
  --benchmark-dir "$BENCH_DIR" \
  --variant ambig_metric \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --agent-bin "$AGENT_BIN" \
  --base-url "$BASE_URL" \
  --skip-existing

# ── Step 5: (optional) Clarify run ────────────────────────────
# Set RUN_CLARIFY=1 to enable the ask-then-solve (clarify) pipeline.
ANSW_MODEL="${ANSWERER_MODEL:-gpt-4o-mini}"
TIMEOUT="${TIMEOUT:-1800}"
ASK_TIMEOUT="${ASK_TIMEOUT:-120}"

if [[ "${RUN_CLARIFY:-0}" == "1" ]]; then
  echo
  echo "[4/5] Running agent on AMBIG_METRIC variant with clarify..."
  "$PY" "$HERE/step_3_run_agent_clarify.py" \
    --benchmark-dir "$BENCH_DIR" \
    --variant ambig_metric \
    --model "$MODEL" \
    --answerer-model "$ANSW_MODEL" \
    --tasks "$TASKS" \
    --agent-bin "$AGENT_BIN" \
    --base-url "$BASE_URL" \
    --timeout "$TIMEOUT" \
    --ask-timeout "$ASK_TIMEOUT" \
    --skip-existing
fi

# ── Step 6: (optional) LLM-judge what metric the agent optimized ──
# Set RUN_JUDGE=1 to enable the judge audit.
N_JUDGES="${N_JUDGES:-1}"

if [[ "${RUN_JUDGE:-0}" == "1" ]]; then
  echo
  echo "[5/5] Judging agent optimization targets (n_judges=$N_JUDGES)..."
  CONDITIONS="full,ambig_metric"
  if [[ "${RUN_CLARIFY:-0}" == "1" ]]; then
    CONDITIONS="full,ambig_metric,ambig_metric+clarify"
  fi
  "$PY" "$HERE/step_4_judge_audit.py" \
    --benchmark-dir "$BENCH_DIR" \
    --judge-model "$MODEL" \
    --agent-models "$MODEL" \
    --conditions "$CONDITIONS" \
    --n-judges "$N_JUDGES"
fi

echo
echo "=========================================================="
echo "Pipeline done."
echo "Results in: $BENCH_DIR/results/"
echo "=========================================================="
