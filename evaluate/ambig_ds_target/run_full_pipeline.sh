#!/usr/bin/env bash
# Full ambig_ds_target pipeline: rebuild ambig CSVs for ALL tasks,
# then run decoy-quality (step 2) and inferability (step 3) audits.
#
# Usage:
#   ./run_full_pipeline.sh                 # all 53 tasks
#   FORCE=1 ./run_full_pipeline.sh         # delete ambig dirs first
#   SKIP_LLM=0 ./run_full_pipeline.sh      # also run LLM selectors D/E/F
#   ./run_full_pipeline.sh task1,task2     # subset by slug
#
# Designed to be wrapped in `caffeinate -is` for overnight runs.

set -euo pipefail

# ---- locate repo + venv ---------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

if [[ -d .venv ]]; then
    source .venv/bin/activate
fi
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# ---- env required by the upstream copy step ------------------------------
export DSBENCH_DATA_ROOT="${DSBENCH_DATA_ROOT:-$HOME/DSBench/data_modeling/data}"
export DSBENCH_PERF_ROOT="${DSBENCH_PERF_ROOT:-$HOME/DSBench/data_modeling/save_performance}"

# lightgbm needs libomp.dylib on macOS without Homebrew
if [[ -f "$REPO/.venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib" ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH="$REPO/.venv/lib/python3.12/site-packages/sklearn/.dylibs:${DYLD_FALLBACK_LIBRARY_PATH:-}"
fi

cd "$HERE"
TASKS_ARG="${1:-}"
LOG_DIR="$HERE/benchmark/_pipeline_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[$(date)] starting full pipeline -> $LOG_DIR/run_${TS}_*.log"

# ---- optional: force rebuild by deleting existing ambig dirs --------------
if [[ "${FORCE:-0}" == "1" ]]; then
    echo "[FORCE] deleting existing ambig dirs..."
    if [[ -n "$TASKS_ARG" ]]; then
        for s in ${TASKS_ARG//,/ }; do
            rm -rf "benchmark/data/$s/ambig"
        done
    else
        rm -rf benchmark/data/*/ambig benchmark/audits
    fi
fi

# ---- step 1: setup / rebuild ---------------------------------------------
S1_LOG="$LOG_DIR/run_${TS}_step1.log"
echo "[$(date)] step 1 -> $S1_LOG"
if [[ -n "$TASKS_ARG" ]]; then
    python -u step_1_setup_benchmark.py --benchmark-dir ./benchmark \
        --tasks "$TASKS_ARG" 2>&1 | tee "$S1_LOG"
else
    python -u step_1_setup_benchmark.py --benchmark-dir ./benchmark \
        2>&1 | tee "$S1_LOG"
fi

# ---- step 2: decoy quality audit -----------------------------------------
S2_LOG="$LOG_DIR/run_${TS}_step2.log"
echo "[$(date)] step 2 -> $S2_LOG"
S2_ARGS=(--benchmark-dir ./benchmark --max_train_rows 5000000)
if [[ -n "$TASKS_ARG" ]]; then
    S2_ARGS+=(--only "$TASKS_ARG")
fi
python -u step_2_decoy_quality.py "${S2_ARGS[@]}" 2>&1 | tee "$S2_LOG"

# ---- step 3: inferability audit ------------------------------------------
S3_LOG="$LOG_DIR/run_${TS}_step3.log"
echo "[$(date)] step 3 -> $S3_LOG"
S3_ARGS=(--benchmark-dir ./benchmark)
if [[ "${SKIP_LLM:-1}" == "1" ]]; then
    S3_ARGS+=(--skip_llm)
fi
if [[ -n "$TASKS_ARG" ]]; then
    S3_ARGS+=(--only "$TASKS_ARG")
fi
python -u step_3_inferability_audit.py "${S3_ARGS[@]}" 2>&1 | tee "$S3_LOG"

echo "[$(date)] DONE."
echo "Summaries:"
echo "  $HERE/benchmark/audits/decoy_quality/decoy_quality_summary.csv"
echo "  $HERE/benchmark/audits/inferability/inferability_audit_summary.csv"
echo "  $HERE/benchmark/audits/inferability/tex/target_inferability_table.tex"
