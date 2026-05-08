#!/usr/bin/env bash
# End-to-end pipeline for the Ambig-DS-M (metric ambiguity) benchmark.
#
# This script demonstrates the full workflow:
#   1. Setup    : download HF prompts + Kaggle competition data
#   2. Generate : (optional) re-generate ambig_metric.md prompts with an LLM
#   3. Audit    : verify all redacted prompts are clean of metric leaks
#   4. Run      : execute an LLM agent on full + ambig_metric variants
#   5. Judge    : (optional) LLM-classify what metric the agent optimized
#
# Usage:
#   ./run_pipeline.sh <benchmark-dir> <model> [tasks]
# Example:
#   ./run_pipeline.sh ./benchmark gpt-4o aerial-cactus-identification,dog-breed-identification
#
# Environment:
#   OPENAI_API_KEY   required
#   OPENAI_BASE_URL  optional (defaults to https://api.openai.com/v1)
#   AGENT            optional, 'claw' (default) or 'opencode'
#   AGENT_BIN        optional path to the agent binary (default: $AGENT on PATH)

set -euo pipefail

BENCH_DIR="${1:-./benchmark}"
MODEL="${2:-gpt-4o-mini}"
TASKS="${3:-all}"

AGENT="${AGENT:-claw}"
AGENT_BIN="${AGENT_BIN:-$AGENT}"
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
echo "  agent         : $AGENT ($AGENT_BIN)"
echo "  base url      : $BASE_URL"
echo "=========================================================="

# ── Step 1: Setup ─────────────────────────────────────────────
echo
echo "[1/5] Setup: downloading prompts (HF) + competition data (Kaggle)..."
SETUP_ARGS=(--benchmark-dir "$BENCH_DIR")
if [[ "$TASKS" != "all" ]]; then
  SETUP_ARGS+=(--tasks "$TASKS")
fi
"$PY" "$HERE/step_1_setup_benchmark.py" "${SETUP_ARGS[@]}"

# ── Step 2: (optional) Re-generate ambig prompts with an LLM ──
# By default we use the prompts shipped on HF. Uncomment to regenerate.
#
# echo
# echo "[2/5] Generating ambiguous prompts with LLM..."
# "$PY" "$HERE/../../create_datasets/ambig_ds_metric/pipeline/step_1_generate_ambig_prompts.py" --benchmark-dir "$BENCH_DIR" --run

# ── Step 3: Audit redacted prompts ────────────────────────────
echo
echo "[3/5] Auditing prompts for metric leaks..."
"$PY" "$HERE/../../create_datasets/ambig_ds_metric/pipeline/step_2_audit_prompts.py" --benchmark-dir "$BENCH_DIR" || {
  echo "WARNING: audit found issues. Continue? (Ctrl-C to abort, Enter to proceed)"
  read -r _
}

# ── Step 4: Run agent on both variants ────────────────────────
echo
echo "[4/5] Running agent on FULL variant..."
"$PY" "$HERE/step_2_run_agent.py" \
  --benchmark-dir "$BENCH_DIR" \
  --variant full \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --agent "$AGENT" \
  --agent-bin "$AGENT_BIN" \
  --base-url "$BASE_URL" \
  --skip-existing

echo
echo "[4/5] Running agent on AMBIG_METRIC variant..."
"$PY" "$HERE/step_2_run_agent.py" \
  --benchmark-dir "$BENCH_DIR" \
  --variant ambig_metric \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --agent "$AGENT" \
  --agent-bin "$AGENT_BIN" \
  --base-url "$BASE_URL" \
  --skip-existing

# ── Step 5: (optional) LLM-judge what metric the agent optimized ──
# Uncomment to run the judge audit.
#
# echo
# echo "[5/5] Judging agent optimization targets..."
# "$PY" "$HERE/step_4_judge_audit.py" \
#   --benchmark-dir "$BENCH_DIR" \
#   --judge-model "$MODEL" \
#   --agent-models "$MODEL" \
#   --conditions full,ambig_metric

echo
echo "=========================================================="
echo "Pipeline done."
echo "Results in: $BENCH_DIR/results/"
echo "=========================================================="
