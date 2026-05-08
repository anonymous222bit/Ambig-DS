# Ambig-DS-M — Evaluation Pipeline

This directory runs ML-engineering coding agents on the **Ambig-DS-M**
benchmark (61 Kaggle competitions × {`full`, `ambig_metric`} prompts) and
grades their submissions against the held-out ground truth via
[MLE-bench](https://github.com/openai/mle-bench).

If you want to (re)build the dataset itself, see
[`create_datasets/ambig_ds_metric/README.md`](../../create_datasets/ambig_ds_metric/README.md).

---

## Layout

```
evaluate/ambig_ds_metric/
├── README.md                            ← this file
├── _llm_client.py                         Shared OpenAI-compatible chat client
├── agents.py                              Agent adapters: claw + opencode (uniform interface)
├── opencode.json                          Reference opencode config (env-var driven)
├── clarify_answerer.py                    Answerer LLM for the clarify protocol
├── grade_submission.py                    Standalone grading utility (single + batch)
├── run_pipeline.sh                        End-to-end orchestrator
│
├── step_1_setup_benchmark.py              Download HF prompts + run mlebench prepare
├── step_2_run_agent.py                    Build workspace → run agent → locate sub → grade
├── step_3_run_agent_clarify.py            Same, but with ASK→ANSWER→SOLVE clarify protocol
└── step_4_judge_audit.py                  LLM judge classifying agent's optimisation target
```

---

## Data flow & filesystem layout

A single user-chosen **benchmark directory** (`<bench>`, e.g.
`./benchmark`) is the only persistent state. Every step reads and writes
underneath it.

```
<bench>/                                         ← created by Step 1
├── README.md                                      from HuggingFace
├── task_list.txt                                  from HuggingFace
├── metric_manifest.json                           from HuggingFace (true metric metadata)
├── metrics_classified.csv                         from HuggingFace
├── edits_log.md                                   from HuggingFace
├── prompts/<slug>/{full,ambig_metric}.md          from HuggingFace
│
├── data/<slug>/prepared/                          from `mlebench prepare` (Kaggle)
│   ├── public/                                       what the agent sees
│   └── private/                                      held-out ground truth, used by the grader
│
├── workspaces/<run-name>/<slug>/                  written by Step 2/3 (per-task scratch)
│   ├── data/                                         symlinks into `<bench>/data/<slug>/prepared/public`
│   ├── task.md                                       prompt + submission instructions
│   ├── _meta.json                                    workspace provenance
│   └── opencode.json                                 (opencode only) ephemeral provider config
│
└── results/<run-name>/<slug>/                     written by Step 2/3 (graded artefacts)
    ├── _submission.csv                               agent's submission (copied from workspace)
    ├── _shape.json                                   shape diagnostics
    ├── _grade.json                                   MLE-bench grading report
    ├── _traj.json                                    agent trajectory + tool uses + cost
    └── _audit.<judge_model>.json                     written by Step 4 (LLM judge label)
results/<run-name>/_runlog.jsonl                   per-task log (one JSON per line)
```

`<run-name>` defaults to `<agent>_<model>_<variant>` (e.g.
`opencode_anthropic_claude_haiku_4_5_v1_0_full`) and can be overridden
with `--run-name`.

---

## Prerequisites

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai>=1.0 huggingface_hub>=0.24 pandas>=2.0 py7zr
pip install -e git+https://github.com/openai/mle-bench.git#egg=mlebench
```

### 2. Kaggle credentials

```bash
mkdir -p ~/.kaggle && cp <your-kaggle.json> ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

You must accept each competition's rules on kaggle.com before
`mlebench prepare` will download its data (otherwise it returns 403).

### 3. LLM API access

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional; any compatible gateway works
```

### 4. Coding-agent CLI

Pick one with `--agent`:

| `--agent`   | Install                                                                   | Default `--agent-bin` |
| ----------- | ------------------------------------------------------------------------- | --------------------- |
| `claw`      | (internal / external `claw` build)                                        | `claw`                |
| `opencode`  | `npm i -g opencode-ai@latest` *or* `curl -fsSL https://opencode.ai/install \| bash` | `opencode`            |

Both adapters live in [`agents.py`](agents.py) and obey the same contract:
`run_agent(agent, bin, model, prompt, cwd, api_key, base_url, timeout) → (message, tool_uses, iters, cost)`.
Adding a third agent is a one-function patch.

For `opencode`, an ephemeral `opencode.json` is written into each
workspace registering an OpenAI-compatible provider whose `apiKey` and
`baseURL` are filled from `{env:OPENAI_API_KEY}` / `{env:OPENAI_BASE_URL}`.
Nothing about the gateway is hard-coded.

---

## Steps

### Step 1 — `step_1_setup_benchmark.py`

```bash
python step_1_setup_benchmark.py --benchmark-dir ./benchmark
```

| Flag                  | Effect                                                                  |
| --------------------- | ----------------------------------------------------------------------- |
| `--skip-data`         | Download only the prompts (~1.5 MB). Useful for inspection.             |
| `--tasks slug1,slug2` | Only prepare these slugs (saves disk).                                  |
| `--verify-only`       | Re-run verification without downloading.                                |
| `--hf-repo other/x`   | Pull from a fork of the HF dataset.                                     |
| `--keep-all-82`       | Disable the 61-task eval-scope filter; keep all tasks from HF.          |

**Reads:** the HuggingFace dataset + Kaggle (per-task).
**Writes:** `<bench>/{prompts,data,task_list.txt,metric_manifest.json,…}`.

#### Eval scope (61 tasks)

After HF download, `task_list.txt` is rewritten to the 61-task evaluation
scope. The excluded slugs are listed in `EXCLUDED_TASKS` in
`step_1_setup_benchmark.py`. Pass `--keep-all-82` to disable the filter.

Downstream steps (2/3/4) read `task_list.txt`, so they automatically
operate on the 61 without any extra flag.

> **Disk note.** Full 61-task Kaggle data is ~63 GB downloaded and
> ~150–200 GB after `mlebench prepare` extracts and resplits.
> For smoke tests, prepare only a few small tasks
> (`random-acts-of-pizza` ~5 MB, `aerial-cactus-identification` ~12 MB,
> `spooky-author-identification` ~2.3 MB).

### Step 2 — `step_2_run_agent.py`

```bash
python step_2_run_agent.py \
    --benchmark-dir ./benchmark \
    --variant {full|ambig_metric} \
    --model <model-id> \
    --agent {claw|opencode} \
    --tasks {all|slug1,slug2}
```

Per task:

1. Build `<bench>/workspaces/<run>/<slug>/` with data symlinked under
   `./data/` and `task.md` containing the chosen prompt + a
   submission-instruction footer telling the agent to write
   `_submission.csv` at the workspace root.
2. Invoke the agent with full filesystem access (`--dangerously-skip-permissions` for opencode).
3. Locate the submission CSV (`_submission.csv` / `submission.csv` / single `*.csv`).
4. Compute shape diagnostics (`_shape.json`).
5. Grade against the private split (`_grade.json`).

**Reads:** `<bench>/prompts/<slug>/<variant>.md`, `<bench>/data/<slug>/prepared/public/`.
**Writes:** `<bench>/workspaces/<run>/<slug>/`, `<bench>/results/<run>/<slug>/{_submission,_shape,_grade,_traj}.{csv,json}`,
`<bench>/results/<run>/_runlog.jsonl`.

Useful flags: `--timeout 600`, `--skip-existing`, `--dry-run`,
`--run-name <custom>`, `--agent-bin /path/to/bin`.

### Step 3 — `step_3_run_agent_clarify.py`

Three-phase variant of Step 2:

1. **ASK** in `<bench>/workspaces/<run>/_ask/<slug>/`: agent sees the
   prompt + clarify protocol and may write a single question to
   `_question.txt` (or `NONE`).
2. **ANSWER** via `clarify_answerer.py`: an LLM answers using the true
   `metric_manifest.json` entry, refusing out-of-scope questions.
3. **SOLVE** in `<bench>/workspaces/<run>/<slug>/`: a fresh workspace,
   prompt augmented with the `[Q, A]` transcript; runs to completion
   like Step 2.

```bash
python step_3_run_agent_clarify.py \
    --benchmark-dir ./benchmark \
    --variant ambig_metric \
    --model <agent-model-id> \
    --answerer-model <answerer-model-id> \
    --agent {claw|opencode}
```

**Default `<run>`:** `<agent>_<model>_<variant>_clarify`
(`+_strict` if `--strict-protocol`, or `<agent>_<model>_<variant>_ask_only`
when `--clarify-only`).

#### Ask policies (`--strict-protocol`)

Two ask policies are supported, mirroring the paper:

- **Permissive** (default): `CLARIFY_PROTOCOL` — agent *may* ask one question.
- **Conservative** (`--strict-protocol`): `STRICT_CLARIFY_PROTOCOL` — agent
  asks only if the task cannot be solved from prompt + data; unnecessary
  clarification is penalized. Verbatim from the paper's
  `run_metric_ambig_mle_clarify.py`.

#### Clarify-only mode (`--clarify-only`)

Run **only** Phase A (ask) + Phase B (answer). Skips Phase C (solve), so no
submission is produced and no grading happens. Mirrors the paper's
`run_metric_ambig_mle_ask_only.py`. Output per slug: `_clarify.json` only.

#### Run all 4 ask conditions (clarify-only)

Runs the four (variant × policy) cells used for the ask-policy sensitivity
analysis: `{full, ambig_metric} × {permissive, conservative}`. Default run
names keep them in separate result dirs so they don't collide:

```bash
MODEL=<agent-model-id>
ANSW=anthropic_claude_haiku_4_5_v1_0

for VARIANT in full ambig_metric; do
  for POLICY in "" "--strict-protocol"; do
    python step_3_run_agent_clarify.py \
        --benchmark-dir ./benchmark \
        --variant "$VARIANT" \
        --model "$MODEL" \
        --answerer-model "$ANSW" \
        --agent claw \
        --clarify-only $POLICY \
        --skip-existing
  done
done
```

Result directories produced:

| variant       | policy        | run dir                                                      |
| ------------- | ------------- | ------------------------------------------------------------ |
| full          | permissive    | `results/claw_<MODEL>_full_ask_only/`                        |
| full          | conservative  | `results/claw_<MODEL>_full_ask_only_strict/`                 |
| ambig_metric  | permissive    | `results/claw_<MODEL>_ambig_metric_ask_only/`                |
| ambig_metric  | conservative  | `results/claw_<MODEL>_ambig_metric_ask_only_strict/`         |

Existing `*_clarify/` runs (full ask→answer→solve) are **not** overwritten:
clarify-only writes to `*_ask_only*` directories.

### Step 4 — `step_4_judge_audit.py`

LLM-classify what each agent run actually optimised. Labels:
`Intended | FormBroken | WrongObjective | Abdicated | Invalid | Other`.

```bash
python step_4_judge_audit.py \
    --benchmark-dir ./benchmark \
    --judge-model <judge-model-id> \
    --agent-models <agent-model-id-1>,<agent-model-id-2> \
    --conditions full,ambig_metric \
    --concurrency 4
```

**Reads:** `<bench>/results/<run>/<slug>/{_grade,_traj,_submission}.{json,csv}`,
`<bench>/metric_manifest.json`.
**Writes:** `<bench>/results/<run>/<slug>/_audit.<judge_model>.json`.

### Standalone grading — `grade_submission.py`

```bash
# Single submission
python grade_submission.py --benchmark-dir ./benchmark \
    --slug <slug> --submission ./my_submission.csv

# Batch grade an entire run
python grade_submission.py --benchmark-dir ./benchmark \
    --results-dir ./benchmark/results/<run-name>
```

### Score normalization to [0, 1] — `normalize_scores.py`

The paper reports `S_full`, `S_ambig`, `S_ask` **normalized to [0, 1]**
before computing $\Delta_{\text{ambig}} = S_{\text{ambig}} - S_{\text{full}}$
and $\Delta_{\text{ask}} = S_{\text{ask}} - S_{\text{ambig}}$. The pipeline
stores raw competition scores in `_grade.json`; this helper rescales each
one to its **leaderboard rank-percentile** (1.0 = beat every Kaggle team,
0.0 = beat none, 0.5 ≈ median), which is direction-safe (uses
`is_lower_better`) and bounded by construction.

```bash
# Annotate every _grade.json under one run dir with a `score_norm_01` field
python normalize_scores.py --benchmark-dir ./benchmark \
    --results-dir ./benchmark/results/opencode_<model>_full

# Wide table across Full / Ambig / Clarify for one (agent, model)
python normalize_scores.py --benchmark-dir ./benchmark \
    --results-root ./benchmark/results \
    --agent opencode --model <model-id> \
    --variants full,ambig_metric,ambig_metric_clarify \
    --out ./benchmark/results/_normalized_<model>.csv
```

Verified result on the smoke test (n=1, `spooky-author-identification`):

| variant | raw score (log loss ↓) | `S_norm_01` (rank-pct ↑) |
|---|---|---|
| `full` | 0.64553 | 0.2504 |
| `ambig_metric` | 0.57341 | 0.2955 |
| `ambig_metric_clarify` | 0.59990 | 0.2746 |

---

## End-to-end orchestrator

```bash
# Defaults: AGENT=claw, AGENT_BIN=$AGENT, BASE_URL=$OPENAI_BASE_URL or OpenAI
./run_pipeline.sh ./benchmark <model-id> <slug-or-all>

# With opencode + a custom binary path:
AGENT=opencode AGENT_BIN=/path/to/opencode \
    ./run_pipeline.sh ./benchmark <model-id> <slug>
```

The orchestrator runs Steps 1, 2 (audit, via the create-side script), 3
(`full` then `ambig_metric` agent runs). Steps 1.5 (regenerate ambig
prompts) and 4 (judge) are commented out by default and must be
uncommented to enable.

---

## Quick smoke test (verified end-to-end, opencode + gemini_3_flash)

Below is the exact recipe that has been run end-to-end on
`spooky-author-identification` (~2 MB Kaggle data, multi-class log loss).

### Prereqs (one-time)

```bash
# Python env
source .venv/bin/activate

# OpenAI-compatible gateway creds (any provider works)
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...

# opencode CLI (per-user install; system /usr/lib is read-only on shared boxes)
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH        # add this to ~/.bashrc to persist
npm install -g opencode-ai
opencode --version                          # sanity check

# Kaggle credentials + accept competition rules on kaggle.com
mkdir -p ~/.kaggle && cp <your-kaggle.json> ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# mle-bench (the CLI is `mlebench`, NOT `python -m mlebench.prepare`)
pip install -e git+https://github.com/openai/mle-bench.git#egg=mlebench
which mlebench                              # must resolve
```

### 1. Download prompts + Kaggle data for one task

```bash
BENCH=/path/to/benchmark            # any writable dir; will be created
TASK=spooky-author-identification

python step_1_setup_benchmark.py --benchmark-dir "$BENCH" --tasks "$TASK"
```

You should see `OK` for the slug and `prepared/{public,private}/`
populated under `$BENCH/data/$TASK/`.

> **Note.** The verifier at the end of step 1 reports `MISSING DATA` for
> all 61 slugs in `task_list.txt` that you didn't request. Harmless —
> only the slugs you `--tasks`-ed are needed.

### 2. Run the agent on both prompt variants

```bash
for V in full ambig_metric; do
  python step_2_run_agent.py \
    --benchmark-dir "$BENCH" \
    --variant "$V" \
    --model gemini_3_flash \
    --tasks "$TASK" \
    --agent opencode \
    --agent-bin "$(command -v opencode)" \
    --base-url "$OPENAI_BASE_URL" \
    --timeout 900
done
```

Per-variant artefacts land in:

```
$BENCH/results/opencode_gemini_3_flash_<variant>/<TASK>/
  ├── _submission.csv     # the agent's CSV
  ├── _grade.json         # mle-bench score + medal thresholds
  ├── _shape.json         # row/col vs sample_submission
  └── _traj.json          # tool calls, elapsed, summary
```

`$BENCH/results/opencode_gemini_3_flash_<variant>/_runlog.jsonl` carries
one line per task per run.

### 3. (Optional) LLM judge — what metric did the agent optimise?

```bash
python step_4_judge_audit.py \
  --benchmark-dir "$BENCH" \
  --judge-model gemini_3_flash \
  --agent-models gemini_3_flash \
  --agent-prefix opencode \
  --conditions full,ambig_metric \
  --only-tasks "$TASK"
```

Writes `_audit.<judge_model>.json` next to each `_grade.json`.
`label ∈ {Intended, FormBroken, WrongObjective, Abdicated, Invalid, Other}`.

### 4. Read scores

```bash
python - <<'PY'
import json, glob, os
BENCH = os.environ["BENCH"]
for v in ("full", "ambig_metric"):
    print(f"\n=== {v} ===")
    for f in sorted(glob.glob(f"{BENCH}/results/opencode_gemini_3_flash_{v}/*/_grade.json")):
        slug = os.path.basename(os.path.dirname(f))
        g = json.load(open(f))
        a_path = f.replace("_grade.json", "_audit.gemini_3_flash.json")
        a = json.load(open(a_path)) if os.path.exists(a_path) else {}
        label = (a.get("judge_parsed") or {}).get("label", "—")
        print(f"  {slug:50s} score={g['score']!s:>10}  "
              f"valid={g['valid_submission']}  above_median={g['above_median']}  "
              f"label={label}")
PY
```

### Troubleshooting

| symptom | fix |
|---|---|
| `No module named mlebench.__main__` | use the `mlebench` console-script (already fixed in step 1). |
| `EACCES … /usr/lib/node_modules/opencode-ai` on `npm install -g` | use a per-user prefix (`npm config set prefix ~/.npm-global` + add `~/.npm-global/bin` to `PATH`). |
| `403` from Kaggle during `mlebench prepare` | log in to kaggle.com and accept the competition's rules. |
| `step_4_judge_audit.py` prints `(no_cell)` for every task | pass `--agent-prefix opencode` (default `opencode`); the run-dir is `<agent>_<model>_<variant>`. |
| `[SSL: CERTIFICATE_VERIFY_FAILED]` from kaggle / pip / huggingface | corporate TLS proxy (e.g. Zscaler) is intercepting. Append the proxy's root CA to certifi's bundle: `security find-certificate -a -p -c "Zscaler" /Library/Keychains/System.keychain >> "$(python -c 'import certifi;print(certifi.where())')"`. |
| `PermissionError: Kaggle authentication failed` | make sure `~/.kaggle/kaggle.json` exists and has mode 600 (`chmod 600 ~/.kaggle/kaggle.json`); see [Kaggle credentials](#kaggle-credentials) below. |
| `_grade.json: AssertionError: Leaderboard must have a 'score' column.` | mle-bench ships `leaderboard.csv` files via Git LFS; if `git lfs` was missing or blocked when `pip install -e mle-bench` ran, the files are still LFS pointers. Fix with the helper in [Step 1.5 below](#step-15--fetch-mle-bench-leaderboards-git-lfs). |
| `step_2_run_agent.py … summary: "ERROR: [Errno 13] Permission denied: ''"` | `--agent-bin` resolved to an empty path. Pass an absolute path: `--agent-bin /Users/$USER/.npm-global/bin/opencode`. |

---

## Verified setup (May 2026, macOS arm64) {#verified-setup}

The following recipe was used end-to-end to evaluate `gemini_3_flash` on
`spooky-author-identification`.

### 0.1 — Python 3.12 venv

mle-bench requires Python ≥3.11. macOS ships with 3.9. Two options:

```bash
# Option A: uv (preferred when network allows)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv

# Option B: standalone build (works behind corporate proxies)
curl -L -o /tmp/python.tar.gz \
  'https://github.com/astral-sh/python-build-standalone/releases/download/20260504/cpython-3.12.13+20260504-aarch64-apple-darwin-install_only_stripped.tar.gz'
mkdir -p ~/.local/python312 && tar xzf /tmp/python.tar.gz -C ~/.local/python312
~/.local/python312/python/bin/python3 -m venv .venv
```

Then:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install 'openai>=1.0' 'huggingface_hub>=0.24' 'pandas>=2.0' py7zr
pip install -e 'git+https://github.com/openai/mle-bench.git#egg=mlebench'
```

### 0.2 — Corporate TLS proxy (Zscaler etc.) {#corporate-tls}

If `kaggle competitions list` returns `[SSL: CERTIFICATE_VERIFY_FAILED]`,
your traffic is being intercepted by a corporate MITM proxy. Append the
proxy's root CA to certifi's bundle:

```bash
security find-certificate -a -p -c "Zscaler" /Library/Keychains/System.keychain \
  >> "$(python -c 'import certifi; print(certifi.where())')"
```

(Substitute the proxy name if not Zscaler. Re-run after every
`pip install certifi` upgrade.)

### 0.3 — Kaggle credentials {#kaggle-credentials}

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{"username":"<your-kaggle-username>","key":"<your-kaggle-key>"}
JSON
chmod 600 ~/.kaggle/kaggle.json

# sanity check
kaggle competitions list | head -3
```

Get the username + key from kaggle.com → Account → "Create New API Token"
(downloads a `kaggle.json`). Then **accept the competition's rules in a
browser** (e.g. <https://www.kaggle.com/c/spooky-author-identification/rules>)
before running `mlebench prepare`, otherwise it returns 403.

### 0.4 — opencode CLI

```bash
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH         # add to ~/.zshrc to persist
npm install -g opencode-ai
opencode --version                           # 1.14.39 verified
```

### 0.5 — Project `.env`

The repo root ships [`.env.example`](../../.env.example). Copy and fill in:

```bash
cd <repo-root>
cp .env.example .env
$EDITOR .env             # set OPENAI_API_KEY, OPENAI_BASE_URL, etc.
set -a && source .env && set +a
```

### Step 1.5 — fetch mle-bench leaderboards (Git LFS) {#step-15--fetch-mle-bench-leaderboards-git-lfs}

`mle-bench` distributes per-competition `leaderboard.csv` files via Git
LFS. They are required by Step 2's grader (otherwise you get
`AssertionError: Leaderboard must have a 'score' column.`). If `git lfs`
was unavailable when `pip install -e mle-bench` ran, the files in
`.venv/src/mlebench/mlebench/competitions/<slug>/leaderboard.csv` are
still ~130-byte LFS pointer stubs.

Use the bundled helper, which downloads every missing CSV from GitHub's
media CDN (same source mle-bench's LFS would use, but no LFS / auth
required):

```bash
python fetch_leaderboards.py                       # fetch every missing leaderboard
python fetch_leaderboards.py --tasks slug1,slug2   # only specific slugs
python fetch_leaderboards.py --force               # re-download everything
```

The script trusts the certifi CA bundle, so it works behind corporate
TLS proxies (e.g. Zscaler) provided you have already appended the
proxy's root CA to certifi (see the Zscaler note above).

If you prefer raw `curl`, the underlying URL pattern is:

```bash
SLUG=spooky-author-identification
LB="$VIRTUAL_ENV/src/mlebench/mlebench/competitions/$SLUG/leaderboard.csv"
curl -sSL -o "$LB" \
  "https://media.githubusercontent.com/media/openai/mle-bench/main/mlebench/competitions/$SLUG/leaderboard.csv"
head -2 "$LB"   # must start with `scoreNullable,teamId,...`
```

### Step 2 invocation that actually works

`opencode` resolved through `$(command -v opencode)` was occasionally
empty in nested shells. Use an absolute path:

```bash
python step_2_run_agent.py \
  --benchmark-dir ./benchmark \
  --variant {full|ambig_metric} \
  --model gemini_3_flash \
  --tasks spooky-author-identification \
  --agent opencode \
  --agent-bin "$HOME/.npm-global/bin/opencode" \
  --base-url "$OPENAI_BASE_URL" \
  --timeout 1200
```

### Verified result

| variant | slug | score (log-loss, lower is better) | valid | label |
|---|---|---|---|---|
| `full` | `spooky-author-identification` | 0.64553 | ✓ | Intended |
| `ambig_metric` | `spooky-author-identification` | 0.57341 | ✓ | Intended |

