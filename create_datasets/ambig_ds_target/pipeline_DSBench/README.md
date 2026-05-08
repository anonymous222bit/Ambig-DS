# pipeline_DSBench: extending the **original DSBench** tabular slice with target ambiguity

This sub-pipeline converts the **DSBench tabular tasks** (which already ship
with a clean per-task prompt at `Dataset/data_modeling/data/data/task/<slug>.txt`)
into target-ambig variants. It is *separate* from `../pipeline/`, which
targets newly-scraped Kaggle competitions:

| folder              | input                                    | use it when |
|---------------------|------------------------------------------|-------------|
| `../pipeline/`      | raw Kaggle download (you run `kaggle CLI`) | adding a NEW Kaggle competition that DSBench does not yet have |
| `pipeline_DSBench/` | upstream DSBench `data_resplit/<slug>/{train,test}.csv` + `task/<slug>.txt` | converting (or extending) DSBench's existing tabular slice |

This folder is **self-contained**: every script needed end-to-end (decoy
generation, validation, LLM prompt rewrite, audit, HF release) lives here.

---

## Pipeline at a glance

This folder is purely about **producing the HF release**. Decoy-quality
diagnostics and the target-inferability audit live in
[`evaluate/ambig_ds_target/`](../../../evaluate/ambig_ds_target/) (steps 2
and 3 there) — that's where you go after consuming the HF release.

| step | script                              | purpose                                                                                  |
|------|-------------------------------------|------------------------------------------------------------------------------------------|
| 1    | `step_1_generate_decoy.py`          | rank-map + bisected noise calibration → write `train/test/sample_submission.csv` + `_manifest.json` per task |
| 2    | `step_2_generate_ambig_prompts.py`  | LLM rewrite of the upstream DSBench task prompt → `ambig_prompt.txt` (target-ambig variant) |
| 3    | `step_3_audit.py`                   | end-to-end release audit: pipeline integrity, prompt invariants, data invariants, eval scripts, manifests. **Skip for dsbench-only repros** — it expects a `competitions/<slug>/` directory from the `kaggle_2026` wave and crashes with `FileNotFoundError` if absent. |
| 4    | `step_4_build_release.py`           | build the public, HF-shaped `release/` tree (no upload). Output is byte-identical to what gets pushed to HuggingFace and is directly consumable by the evaluator (`evaluate/ambig_ds_target/step_1_setup_benchmark.py --release-source local`). |
| 5    | `step_5_upload_to_hf.py`            | push `release/` to HuggingFace                                                          |

`_llm_client.py` is a tiny shared helper for the OpenAI-compatible chat API;
it is imported by step 2.

---

## What this pipeline produces (and the paper text it implements)

For each task, we construct the decoy in two steps (paper §4.2):

> **(i) Rank-mapping.** Select the 3–10 anonymized features with the lowest
> absolute Spearman correlation to the true target, standardize each, sum
> them per row to get a synthetic score, rank-map the true-target marginal
> onto that score.
> **(ii) Noise calibration.** Per-task **binary-search** the noise level $\eta$
> so that the 3-fold HistGradientBoosting CV gap $|cv_{decoy} - cv_{true}|
> \le 0.02$. Classification: swap a fraction $\eta$ of labels (preserves
> marginal). Regression: add Gaussian noise of std $\eta \cdot \mathrm{std}(y)$
> to the synthetic score and re-rank-map.

`step_1_generate_decoy.py` is the implementation.

---

## Layout assumptions

You need a workspace root containing upstream DSBench and the in-progress
ambig outputs:

```
$AMBIG_DSBENCH_ROOT/
├── Dataset/data_modeling/data/data/
│   ├── data_resplit/<slug>/{train,test,test_answer,sample_submission}.csv  # INPUT
│   └── task/<slug>.txt                                                     # full prompt (input to step 4)
├── DSBench/data_modeling/evaluation/<slug>_eval.py                          # eval script
└── final_data_v3/target_ambig/
    ├── data/<slug>/{train,test,sample_submission}.csv  + _manifest.json    # step 1 output (decoy CSVs)
    │                                                   + ambig_prompt.txt  # step 4 output (local copy)
    └── data_modeling/data/data/
        ├── data_ambig_target_v3_gen/<slug>/...                              # mirror destination
        └── task_ambig_target_v3_gen/<slug>.txt                              # step 4 output (canonical prompt path)
```

Set this once:

```bash
export AMBIG_DSBENCH_ROOT=/path/to/your/workspace
```

---

## Quickstart — end-to-end on a single task (verified working)

The example below regenerates `bike-sharing-demand` from scratch into a
throw-away workspace and produces a release directory whose layout +
schema is byte-identical to the current HuggingFace release. Substitute
your own paths and slug list as needed.

### 0. Set up workspace

```bash
# absolute paths (edit to match your machine)
PROJ=/abs/path/to/info_theory
WORKSPACE=/tmp/ambig_workspace            # fresh scratch dir
ENV_FILE=$PROJ/project_5/.env             # OPENAI_API_KEY + OPENAI_BASE_URL

export AMBIG_DSBENCH_ROOT=$WORKSPACE

# Mirror the upstream DSBench data tree (source: project_6/Dataset/...)
mkdir -p $WORKSPACE
cp -r $PROJ/project_6/Dataset $WORKSPACE/

# Mirror the eval scripts (step 4 looks here)
mkdir -p $WORKSPACE/DSBench/data_modeling/evaluation
cp $PROJ/project_6/DSBench/data_modeling/evaluation/*_eval.py \
   $WORKSPACE/DSBench/data_modeling/evaluation/

# Choose the slug(s) to (re)build
SLUGS="bike-sharing-demand"
```

### 1. Generate calibrated decoy CSVs + manifest

The task spec CSV must contain at minimum a `task` and a `target_name`
column. Concatenate the two source CSVs that ship with v3 if your slug
isn't in either of them individually:

```bash
SRC=$PROJ/project_6/final_data_v3/target_ambig
SPEC=$WORKSPACE/_combined_tasks.csv
{ head -1 $SRC/target_ambiguity_tasks.csv;
  tail -n +2 $SRC/target_ambiguity_tasks.csv;
  tail -n +2 $SRC/expansion_tasks.csv; } > $SPEC

python step_1_generate_decoy.py \
    --tasks_csv      $SPEC \
    --src_data_root  $WORKSPACE/Dataset/data_modeling/data/data/data_resplit \
    --out_root       $WORKSPACE/final_data_v3/target_ambig \
    --only           "$(echo $SLUGS | tr ' ' ',')" \
    --force
```

> **Important**: `--src_data_root` must point at `data_resplit/` (the
> per-slug `train.csv` lives at
> `data_resplit/<slug>/train.csv`). The default is wrong if you only
> have `Dataset/data_modeling/data/data/<slug>/...`.

Outputs (per slug):
- `$WORKSPACE/final_data_v3/target_ambig/data/<slug>/{train,test,sample_submission}.csv`
- `$WORKSPACE/final_data_v3/target_ambig/data/<slug>/_manifest.json`

### 2. Mirror data into the layout step 4 expects

Step 4 reads the per-slug data from
`final_data_v3/target_ambig/data_modeling/data/data/data_ambig_target_v3_gen/<slug>/`.
Step 1 doesn't write there directly, so copy:

```bash
for s in $SLUGS; do
  SRC=$WORKSPACE/final_data_v3/target_ambig/data/$s
  DST=$WORKSPACE/final_data_v3/target_ambig/data_modeling/data/data/data_ambig_target_v3_gen/$s
  mkdir -p $DST
  cp $SRC/{train.csv,test.csv,sample_submission.csv} $DST/
done
```

### 3. Generate the target-ambig prompt (LLM rewrite)

```bash
unset OPENAI_API_KEY OPENAI_BASE_URL                  # avoid stale shell creds
export AMBIG_LLM_MODEL=anthropic_claude_opus_4_7      # any chat-completions model

for s in $SLUGS; do
  python step_2_generate_ambig_prompts.py \
      --slug      $s \
      --env-file  $ENV_FILE \
      --model     anthropic_claude_opus_4_7 \
      --force
done
```

Outputs (mirror copies, both written by step 2):
- `$WORKSPACE/final_data_v3/target_ambig/data/<slug>/ambig_prompt.txt`
- `$WORKSPACE/final_data_v3/target_ambig/data_modeling/data/data/task_ambig_target_v3_gen/<slug>.txt`

The current `SYSTEM_PROMPT` in step 2 **forbids** disclosing the two-target
setup (the agent must spontaneously detect ambiguity). The validator
correspondingly flags any disclosure phrase (`candidate target`,
`decoy`, `ambiguous`, …) and any uncited numeric fact. Warnings printed
to stdout are advisory; the file is still written.

### 4. Build the public release directory

```bash
python step_4_build_release.py \
    --out    $WORKSPACE/release \
    --tasks  "$(echo $SLUGS | tr ' ' ',')"
```

Produces:

```
$WORKSPACE/release/
├── README.md
├── tasks.csv
└── tasks/<slug>/
    ├── task.txt          # clean (non-ambig) prompt
    ├── task_ambig.txt    # target-ambig prompt
    ├── eval.py           # per-task DSBench-style evaluator
    └── _manifest.json    # public schema (schema_version, source, task,
                          #                ambig_recipe, diagnostics, eval)
```

This layout is byte-identical in structure (and bit-identical in `task.txt`
+ `eval.py` + manifest schema) to the current HF release. Verify with:

```bash
diff -rq $WORKSPACE/release $PROJ/_gh_work/tmp/repro_check/hf_release
```

(Per-task `task_ambig.txt` will differ — it's the new Opus-generated
prompt — and `cv_decoy` / `cv_ratio_decoy_over_true` may differ slightly
because step 1's bisection is sensitive to RNG state.)

### 5. (Optional) Upload to HuggingFace

```bash
AMBIG_HF_REPO=your-handle/Ambig-DS-T \
    python step_5_upload_to_hf.py --skip-build --out $WORKSPACE/release
```

(Without `--skip-build`, step 5 rebuilds `release/` first.)

---

## Scaling to many tasks

Replace `SLUGS="bike-sharing-demand"` above with a space-separated list,
e.g.:

```bash
SLUGS="bike-sharing-demand cat-in-the-dat cat-in-the-dat-ii \
       commonlitreadabilityprize dont-overfit-ii instant-gratification"
```

Steps 1, 2, 3, and 4 all accept either `--only slug1,slug2` (step 1) or
`--tasks slug1,slug2` (step 4) or `--slug <one>` (step 2 — call it once
per slug in a loop).

---

## Reference: hyperparameters used in the paper

| param                      | value | rationale |
|----------------------------|-------|-----------|
| `--cv_tolerance`           | 0.02  | matches the paper's "within 0.02" bound |
| `--bisection_steps`        | 12    | converges $\eta$ to ~0.0002 precision over [0, 0.5] |
| `--max_noise`              | 0.50  | hard cap; tasks needing more are left at the closest candidate |
| `--noise_classification`   | 0.10  | fallback only (used when calibration cannot run) |
| `--noise_regression`       | 0.10  | fallback only |
| `--pool_max`               | 8     | upper bound on the number of low-Spearman features used to build the synthetic score |
| `--pool_frac`              | 0.6   | take the bottom 60% of features by `|Spearman(feature, y)|` |
| `--apply_dtype_snap`       | on    | snap the noised decoy back to the truth's dtype (e.g. int 0/1) |

Per the paper, **39 of 51** retained tasks meet the 0.02 gap criterion; the
rest fall back to the closest candidate that still passes the marginal-match
and low-correlation filters. Expect a similar pass rate on new tasks.

---

## Why these 6 (and not the other 17 leftover DSBench tasks)

DSBench has 77 tabular tasks with eval scripts; 51 became ambig in the v3
release. Of the remaining 26:

- **6 viable** (these): single-target tabular with usable `data_resplit/`
- **7 NLP**: `feedback-prize-english-language-learning`, `google-quest-challenge`, `learning-agency-lab-automated-essay-scoring-2`, `lmsys-chatbot-arena`, `nlp-getting-started`, `tweet-sentiment-extraction`, `us-patent-phrase-to-phrase-matching` — no synthetic decoy column makes sense for free text
- **7 time-series**: `covid19-global-forecasting-week-{1..5}`, `demand-forecasting-kernels-only`, `liverpool-ion-switching` — target structurally fixed by date/series
- **3 unusual format**: `conways-reverse-game-of-life-2020` (grid), `microsoft-malware-prediction` (extreme sparsity), `see-click-predict-fix` (geo + text)
- **4 multi-target**: `playground-series-s3e18`, `s3e20`-multi-output, `s4e3`, `tabular-playground-series-jul-2021` — would require a multi-decoy extension to the rank-mapping step (future work)

If you adapt this pipeline for the multi-target case, the extension point is
`build_decoy()` in `step_1_generate_decoy.py`.

---

## File reference

| file                               | role |
|------------------------------------|------|
| `step_1_generate_decoy.py`         | rank-mapping + per-task bisected noise calibration + dtype snap |
| `step_2_generate_ambig_prompts.py` | LLM rewrite of upstream DSBench task prompt → target-ambig variant |
| `step_3_audit.py`                  | end-to-end release audit (5 invariants per task) |
| `step_4_build_release.py`          | assemble HF-shaped `release/` tree locally (no upload) |
| `step_5_upload_to_hf.py`           | thin wrapper: rebuild + push `release/` to HuggingFace |
| `_llm_client.py`                   | shared OpenAI-compatible chat helper |
