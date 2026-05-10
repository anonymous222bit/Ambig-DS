# Ambig-DS-T evaluator

Run a coding agent against the **target-ambiguity** benchmark
(`anonymous222bit/Ambig-DS-T` on Hugging Face) and grade with the per-task
DSBench evaluators.

This folder is the *evaluator*. The *creator* lives at
`../../create_datasets/ambig_ds_target/pipeline_DSBench/` and is what built
the HF release in the first place.

---

## Layout (under `<benchmark-dir>`)

After `step_1_setup_benchmark.py`:

```
<benchmark-dir>/
├── release/
│   └── tasks/<slug>/
│       ├── task.txt                # original DSBench prompt
│       ├── task_ambig.txt          # target-ambig rewrite
│       ├── eval.py                 # DSBench grader
│       └── _manifest.json          # decoy provenance
├── data/
│   └── <slug>/
│       ├── full/                   # original DSBench data (train/test/sample_submission/test_answer)
│       └── ambig/                  # regenerated decoy variant (train/test/_manifest.json)
├── baselines/<slug>/{gt.txt, baseline.txt}   # for RPG normalization
├── workspaces/<run-name>/<slug>/   # per-task agent workspace (transient)
├── results/<run-name>/             # per-task outputs + _runlog.jsonl
├── audits/<slug>/                  # steps 2–3 quality / inferability audits
└── task_list.txt
```

Per-task results live at `<benchmark-dir>/results/<run-name>/<slug>/`:
`_submission.csv`, `_shape.json`, `_grade.json`, `_traj.json`, plus
`_clarify.json` for step 5.

---

## Prerequisites

```bash
pip install huggingface_hub pandas openai scikit-learn scipy numpy lightgbm
# install the opencode agent
npm i -g opencode-ai
```

### One-time: clone upstream DSBench for the original task data

The HF release ships only the *transformed* prompts, manifests, and
per-task `eval.py`. The original train/test/answer CSVs **and** the
best-known / baseline scores used for RPG normalization come from upstream
DSBench. The git repo does **not** bundle `data.zip` — fetch it from the
official HF mirror (`liqiang888/DSBench`, ~3.13 GB):

```bash
# 1. Clone the upstream repo (code + readmes only).
git clone --depth 1 https://github.com/liqiangjing/DSBench.git ~/DSBench

# 2. Download the tabular data bundle from HuggingFace into data_modeling/.
#    `huggingface-cli` ships with `huggingface_hub` (all versions).
cd ~/DSBench/data_modeling
huggingface-cli download liqiang888/DSBench --repo-type dataset \
    --include "data_modeling/data.zip" --local-dir .

# 3. Unzip it. The archive expands to ./data/{data_resplit,answers,task}/.
unzip -q data_modeling/data.zip -d .

# 4. Unzip the bundled performance scores (already in the cloned repo).
#    Produces ./save_performance/{GT,baseline,gpt-3.5-turbo-0125}/<slug>/result.txt.
unzip -q -o save_performance.zip

export DSBENCH_DATA_ROOT=~/DSBench/data_modeling/data
export DSBENCH_PERF_ROOT=~/DSBench/data_modeling/save_performance
# Sanity check:
ls "$DSBENCH_DATA_ROOT"/data_resplit | head    # one dir per slug
ls "$DSBENCH_DATA_ROOT"/answers      | head    # holds test_answer.csv
cat "$DSBENCH_PERF_ROOT"/GT/playground-series-s3e17/result.txt   # e.g. 1.0
cat "$DSBENCH_PERF_ROOT"/baseline/playground-series-s3e17/result.txt  # e.g. 0.5
```

`DSBENCH_DATA_ROOT` must contain `data_resplit/<slug>/{train,test,sample_submission}.csv`
and `answers/<slug>/test_answer.csv`. `DSBENCH_PERF_ROOT` must contain
`GT/<slug>/result.txt` (best-known) and `baseline/<slug>/result.txt` (baseline);
step 1 copies these into `<bench>/baselines/<slug>/{gt,baseline}.txt` and
step 4 uses them to compute the Relative Performance Gap (RPG):
$\mathrm{RPG} = \max((p - b)/(g - b), 0)$.

### LLM credentials

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # or any OAI-compatible
```

### Reproducing paper results

The paper's reported numbers use the following model configuration:

| Role | Model | How to set |
|------|-------|------------|
| Prompt generation (create\_datasets) | Claude Opus 4.7 | `AMBIG_LLM_MODEL=anthropic_claude_opus_4_7` in `.env` |
| Clarification oracle (step 5 answerer) | Claude Haiku 4.6 | `--answerer-model anthropic_claude_haiku_4_6` |
| Evaluated agent backbones | Gemini 3 Flash, GPT-5.4 Nano, Claude Haiku 4.5, Gemini 3.1 Pro, GPT-5.4 | `--model <id>` in steps 4/5 |

Code defaults (e.g. `gpt-4o-mini` for generation) are development
conveniences and will **not** reproduce the paper's tables.

---

## Step 1 — set up the benchmark

Installs the release tree (prompts + manifests + per-task `eval.py`) into
`<bench>/release/`, copies upstream DSBench CSVs into `<bench>/data/<slug>/full/`,
re-runs the decoy generator on every task to produce
`<bench>/data/<slug>/ambig/`, and copies upstream DSBench's GT/baseline
scores into `<bench>/baselines/<slug>/{gt,baseline}.txt` for RPG normalization.

There are **two equivalent sources** for the release tree. Pick one:

### Option A — from HuggingFace (default)

```bash
python step_1_setup_benchmark.py \
    --benchmark-dir ./benchmark \
    --release-source hf \
    --hf-repo anonymous222bit/Ambig-DS-T \
    --dsbench-data-root "$DSBENCH_DATA_ROOT" \
    --dsbench-perf-root "$DSBENCH_PERF_ROOT"
```

### Option B — from a locally built pipeline release

Run the creator pipeline yourself and point the evaluator at its output:

```bash
# (one-time) build the public release locally
cd ../../create_datasets/ambig_ds_target/pipeline_DSBench
python step_1_generate_decoy.py        ...   # see that folder's README
python step_2_generate_ambig_prompts.py ...
python step_4_build_release.py --out ./release   # writes ./release/

# back here
cd -
python step_1_setup_benchmark.py \
    --benchmark-dir ./benchmark \
    --release-source local \
    --release-path ../../create_datasets/ambig_ds_target/pipeline_DSBench/release \
    --dsbench-data-root "$DSBENCH_DATA_ROOT" \
    --dsbench-perf-root "$DSBENCH_PERF_ROOT"
```

The two paths produce the same `<bench>/release/` layout; everything from
step 2 onward is identical.

Use `--tasks slug1,slug2` to subset, `--verify-only` to re-check an existing
benchmark dir without rebuilding.

## Steps 2 / 3 (optional) — quality audits

Read-only diagnostics over an already-set-up `<benchmark-dir>`. Both scripts
walk `<bench>/data/<slug>/ambig/{train.csv, _manifest.json}` and write into
`<bench>/audits/`:

```bash
python step_2_decoy_quality.py     --benchmark-dir ./benchmark
python step_3_inferability_audit.py --benchmark-dir ./benchmark
```

(The noise-injection sweep that backs Table 6 is creator-side; see
`../../create_datasets/ambig_ds_target/pipeline_DSBench/validate_noise_ablation.py`.)

## Step 4 — run the agent (no clarification)

```bash
# Full prompt (control)
python step_4_run_agent.py --benchmark-dir ./benchmark \
    --variant full --model anthropic_claude_haiku_4_5_v1_0

# Target-ambiguous prompt
python step_4_run_agent.py --benchmark-dir ./benchmark \
    --variant ambig_target --model anthropic_claude_haiku_4_5_v1_0
```

Useful flags: `--tasks slug1,slug2`,
`--timeout 1800`, `--skip-existing`, `--dry-run`,
`--run-name my_run`, `--agent-bin /path/to/opencode`.

## Step 5 — run the agent WITH one-turn clarification

Three-phase protocol:
**Phase A** (ASK) — agent writes one question to `_question.txt` and stops.
**Phase B** (ANSWER) — answerer LLM, given the per-task `_manifest.json`
(which of `val_1`/`val_2` is the real target), produces a 1–2 sentence reply.
**Phase C** (SOLVE) — agent gets the prompt + the [Q,A] transcript and
produces the submission.

```bash
python step_5_run_agent_clarify.py --benchmark-dir ./benchmark \
    --variant ambig_target \
    --model anthropic_claude_haiku_4_5_v1_0 \
    --answerer-model anthropic_claude_haiku_4_6
```

Flags mirror step 4 plus `--answerer-model` and `--ask-timeout`.

**Default `<run-name>`:** `<agent>_<model>_<variant>_clarify`
(`+_strict` if `--strict-protocol`, or `<agent>_<model>_<variant>_ask_only`
when `--clarify-only`).

### Ask policies (`--strict-protocol`)

Two ask policies are supported, mirroring the paper:

- **Permissive** (default): `CLARIFY_PROTOCOL` — agent *may* ask one question.
- **Conservative** (`--strict-protocol`): `STRICT_CLARIFY_PROTOCOL` — agent
  asks only if the task cannot be solved from prompt + data; unnecessary
  clarification is penalized. Verbatim from the paper's
  original clarify script.

### Clarify-only mode (`--clarify-only`)

Run **only** Phase A (ask) + Phase B (answer). Skips Phase C (solve), so no
submission is produced and no grading happens. Mirrors the paper's
ask-only script. Output per slug: `_clarify.json` only.

### Run all 4 ask conditions (clarify-only)

Runs the four (variant × policy) cells used for the ask-policy sensitivity
analysis: `{full, ambig_target} × {permissive, conservative}`. Default run
names keep them in separate result dirs so they don't collide:

```bash
MODEL=gemini_3_flash
ANSW=anthropic_claude_haiku_4_6

for VARIANT in full ambig_target; do
  for POLICY in "" "--strict-protocol"; do
    python step_5_run_agent_clarify.py \
        --benchmark-dir ./benchmark \
        --variant "$VARIANT" \
        --model "$MODEL" \
        --answerer-model "$ANSW" \
        --agent opencode \
        --clarify-only $POLICY \
        --skip-existing
  done
done
```

Result directories produced:

| variant       | policy        | run dir                                                       |
| ------------- | ------------- | ------------------------------------------------------------- |
| full          | permissive    | `results/opencode_<MODEL>_full_ask_only/`                     |
| full          | conservative  | `results/opencode_<MODEL>_full_ask_only_strict/`              |
| ambig_target  | permissive    | `results/opencode_<MODEL>_ambig_target_ask_only/`             |
| ambig_target  | conservative  | `results/opencode_<MODEL>_ambig_target_ask_only_strict/`      |

Existing `*_clarify/` runs (full ask→answer→solve) are **not** overwritten:
clarify-only writes to `*_ask_only*` directories.

---

## Grading

Each task is graded by its own `release/tasks/<slug>/eval.py` against
`data/<slug>/full/test_answer.csv`. The raw score lands in
`results/<run-name>/<slug>/_grade/<slug>/result.txt` and is mirrored into
`_grade.json` together with `score_rpg`, the Relative Performance Gap
$\max((p-b)/(g-b),0)$ used by the paper for cross-task aggregation
(`b`, `g` come from `<bench>/baselines/<slug>/{baseline,gt}.txt`).

## Step 6 — aggregate paired statistics

After running steps 4 and 5 for the same `<run-prefix>` (typically
`<agent>_<model>`), aggregate the per-task RPG scores into the paper's
headline table — macro-averaged $S_{\text{full}}$, $S_{\text{ambig}}$,
$S_{\text{ask}}$, the paired deltas $\Delta_{\text{ambig}}$ and
$\Delta_{\text{ask}}$, one-sided paired Wilcoxon $p$-values, and paired
bootstrap 95% CIs:

```bash
python step_6_aggregate.py --benchmark-dir ./benchmark \
    --run-prefix opencode_gemini_3_flash
```

Writes `<bench>/results/_aggregate/<run-prefix>.json` and prints a table.
Pass a comma-separated list to `--run-prefix` to aggregate several models
in one call.

## Step 7 — target-framing audit (intended vs alternative)

Independent of the score, classify *which* candidate target column
(`val_1` or `val_2`) the agent actually optimized. Per task:
`intended`, `alternative`, or `invalid`. Truth is read from
`<bench>/data/<slug>/ambig/_manifest.json` (the as-built manifest the
agent actually saw). The classifier scans the agent's per-task workspace
under `<bench>/workspaces/<run-name>/<slug>/` for `y = train['val_X']`,
`target = 'val_X'`, `.pop('val_X')`, `.drop(['val_X', ...])`, etc.

```bash
python step_7_target_audit.py --benchmark-dir ./benchmark \
    --run-name opencode_gemini_3_flash_ambig_target,\
opencode_gemini_3_flash_ambig_target_clarify
```

Writes `<bench>/results/<run-name>/<slug>/_target_audit.json` per task and
`<bench>/results/_aggregate/target_audit/<run-name>.json` per run.
**Workspaces must be present** (don't manually delete `<bench>/workspaces/`
between step 4/5 and this step).

## Notes

- Step 1 imports `process_task` from
  `../../create_datasets/ambig_ds_target/pipeline_DSBench/step_1_generate_decoy.py`
  via `sys.path` so the decoy logic stays single-source.
- The `ambig` variant intentionally omits `sample_submission.csv` because
  the rewritten prompt drops references to it.
- Step 4 and step 5 use opencode by default; pass `--agent-bin` if the binary
  is not on PATH.
