# Ambig-DS-M — Dataset Creation Pipeline

This directory builds the **Ambig-DS-M** dataset: 61 Kaggle competition
prompts each available in two variants — `full` (original) and
`ambig_metric` (evaluation metric redacted) — together with a
`metric_manifest.json` that records the true metric for every task.

(The `prompts/` directory under the working benchmark may contain extra
slugs; the canonical task set is whatever is listed in `task_list.txt`.
Steps 2 and 3 only process slugs in `task_list.txt`.)

The output is a HuggingFace dataset:
[`anonymous222bit/Ambig-DS-M`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-M).

> **You only need this directory if you want to (re)build the dataset.**
> To *evaluate* an agent on the existing dataset, see
> [`evaluate/ambig_ds_metric/README.md`](../../evaluate/ambig_ds_metric/README.md).

---

## Layout

```
create_datasets/ambig_ds_metric/
├── README.md                              ← this file
└── pipeline/
    ├── _llm_client.py                       Shared OpenAI-compatible chat client
    ├── step_1_generate_ambig_prompts.py     Rewrite full → ambig_metric prompts (LLM)
    ├── step_2_llm_verify.py                 LLM judge against the paper's 4-item checklist
    └── step_3_upload_to_hf.py               Stage + push the dataset to HuggingFace
```

---

## Data flow

| Step | Reads from                                                | Writes to                                                                              |
| ---- | --------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1    | `<bench>/prompts/<slug>/full.md` + `metric_manifest.json` | `<bench>/prompts/<slug>/ambig_metric.md`                                               |
| 2    | `<bench>/prompts/<slug>/{full,ambig_metric}.md` + manifest | `<bench>/_verify/<slug>.json`, `_summary.json`, `rejected.txt`                         |
| 3    | `<bench>/` (everything except `data/`) + `_verify/`        | local staging dir + `https://huggingface.co/datasets/<repo-id>` (when `--upload`)      |

`<bench>` is the working benchmark directory created by
`evaluate/ambig_ds_metric/step_1_setup_benchmark.py` (Step 1 of the eval
pipeline pulls the prompts from HuggingFace). From this directory, that
is `../../evaluate/ambig_ds_metric/benchmark`.

---

## Prerequisites

```bash
# Python deps
pip install 'openai>=1.0' 'huggingface_hub>=0.24' 'pandas>=2.0'

# LLM access (any OpenAI-compatible endpoint)
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional

# HuggingFace push token (only for step_3_upload_to_hf.py with --upload)
export HF_TOKEN=hf_...
```

---

## Steps

### Step 1 — `step_1_generate_ambig_prompts.py`

LLM rewrite of every `full.md` into `ambig_metric.md`, removing direct
mention of the evaluation metric while preserving every other detail.

```bash
python pipeline/step_1_generate_ambig_prompts.py \
    --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark \
    --run \
    --model gpt-4o
```

The default model is read from `AMBIG_LLM_MODEL` (falls back to
`gpt-4o-mini`). Output files are written with a trailing newline so
they are byte-identical to the released HuggingFace artifacts.

The per-call output cap is read from `AMBIG_LLM_MAX_TOKENS`
(default `16384`). The redacted prompt is roughly the same length as
`full.md` and some Kaggle prompts exceed 30 KB, so the gateway's
default ~2–4 K cap silently truncates them. Step 1 also refuses to
write a redacted prompt that is shorter than 50 % of `full.md` and
prints a warning so the slug can be re-generated with a higher cap.

| Flag                  | Effect                                                 |
| --------------------- | ------------------------------------------------------ |
| (default, no `--run`) | Dry run — print plan only.                             |
| `--run`               | Actually call the LLM.                                 |
| `--force --slugs A B` | Re-generate the listed slugs even if they already exist. |
| `--manifest-only`     | Generate manifest entries only.                        |
| `--prompts-only`      | Generate prompts only, skip manifest writes.           |

**Output:** `<bench>/prompts/<slug>/ambig_metric.md` for every slug in
`task_list.txt`.

### Step 2 — `step_2_llm_verify.py`

LLM judge that applies the paper's four-item retention checklist
(§3.3, “Verification and Filtering”) to every `ambig_metric.md`:

1. **Plausible alternatives** — ≥2 reasonable metrics remain consistent
   with the redacted prompt and the implied data package.
2. **Ambiguity preserved** — the true metric does not leak (Evaluation
   section, inline mentions, formulas, optimization direction, or
   submission-format hints).
3. **Decision relevant** — resolving the ambiguity changes a task-level
   choice (hard labels vs probabilities, direction, top-K, clipping,
   column-wise aggregation, submission semantics, …).
4. **Task preserved** — only metric-related information was removed.

```bash
python pipeline/step_2_llm_verify.py --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark --run
```

For each slug the judge produces strict JSON with per-check pass/fail,
rationale, the list of validated alternatives, and any quoted leaked
cues. Outputs:

- `<bench>/_verify/<slug>.json` — full verdict
- `<bench>/_verify/_summary.json` — aggregate counts + per-slug verdicts
- `<bench>/_verify/rejected.txt` — slugs that failed any check

Default judge model is `AMBIG_VERIFIER_MODEL` (falls back to
`AMBIG_LLM_MODEL`, then to `gpt-4o-mini`). To approximate the paper's
cross-verifier audit (non-Claude verifier families), re-run with
`--model gpt-4o --out-tag gpt` and `--model gemini-2.5-pro --out-tag
gemini` so the verdicts land under `_verify_gpt/` / `_verify_gemini/`.

Useful flags: `--slugs A B` (subset), `--force` (re-judge even when
`<slug>.json` already exists), `--out-tag <tag>` (alternate output
directory).

> **Note on the HF release:** The `_verify/rejected.txt` shipped with
> [`anonymous222bit/Ambig-DS-M`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-M)
> contains `jigsaw-unintended-bias-in-toxicity-classification`, which
> initially failed LLM verification. A human judge reviewed the prompt,
> confirmed the redaction was correct, and the per-slug verdict was
> updated to `pass` (reflected in `_summary.json`, 61/61 pass).
> `rejected.txt` was not regenerated after that override. The canonical
> source of truth is `_summary.json`.

### Step 3 — `step_3_upload_to_hf.py`

Stage the dataset (everything except `data/`, which is the user-downloaded
Kaggle data) into a clean directory and optionally push it to a HuggingFace
dataset repo.

```bash
# Dry-run (stage only, do not push)
python pipeline/step_3_upload_to_hf.py --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark

# Stage + push
python pipeline/step_3_upload_to_hf.py \
    --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark \
    --upload \
    --repo-id <your-org>/Ambig-DS-M
```

The stager copies (in this order): `task_list.txt`, `metric_manifest.json`
(stripping any `_doc` key, attaching `validated_alternatives` from the
step-2 verifier when present),
`metrics_classified.csv` (optional — included if present),
`edits_log.md` (optional — looked up at `<bench>/edits_log.md` or
`<bench>/prompts/EDITS_LOG.md` for backward compatibility),
`prompts/<slug>/{full,ambig_metric}.md` for each slug in `task_list.txt`,
and a generated `README.md`.

Gating on the verifier output:

- If `<bench>/_verify/_summary.json` exists, slugs with verdict `fail`
  block staging. Pass `--allow-failed` to override (paper's “human
  override” branch) or `--require-verify` to refuse to stage at all
  without a verifier summary.

---

## Reproducing the released HF dataset

To reproduce [`anonymous222bit/Ambig-DS-M`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-M)
byte-identically, use:

```bash
export AMBIG_LLM_MODEL=anthropic_claude_opus_4_7   # model used for the release
python pipeline/step_1_generate_ambig_prompts.py --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark --run
python pipeline/step_2_llm_verify.py               --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark --run
python pipeline/step_3_upload_to_hf.py             --benchmark-dir ../../evaluate/ambig_ds_metric/benchmark
```

Two LLM-determined details that may vary run-to-run (sampler / model drift):
minor wording in the neutralized "predict a value …" sentence, and rare
subsection rewordings. Re-running on the same model usually reproduces the
HF artifacts within a few characters.

---

## Citation

```bibtex
@article{ambig-ds-m-2026,
  title = {Ambig-DS-M: Measuring the Impact of Metric Ambiguity on ML Engineering Agents},
  year  = {2026}
}
```

## License

Code: MIT. Prompts are derivative works of publicly available Kaggle
competition descriptions, redistributed following the precedent set by
[MLE-bench](https://github.com/openai/mle-bench) (MIT). The underlying
Kaggle datasets must be downloaded separately and are subject to each
competition's own rules.
