# Ambig-DS-T: Target-Ambiguity Benchmark for Tabular Data Science

> **Paper ↔ code naming:** The paper uses *Ambig-DS-Target*; the code
> and HuggingFace use *Ambig-DS-T*.

`Ambig-DS-T` is a benchmark for evaluating how language models behave under
**target ambiguity** — when the prompt does not unambiguously identify which
column of a tabular dataset to predict. Every task ships in two arms:

| arm        | prompt           | data                                      |
|------------|------------------|-------------------------------------------|
| `full`     | `task.txt`       | clean Kaggle CSVs (one true target)       |
| `ambig`    | `task_ambig.txt` | CSVs with **decoy** columns injected that are statistically plausible alternative targets |

A robust model should (a) ask a clarifying question on `ambig`, or (b) at
minimum, score no better than a strong heuristic baseline that picks a decoy.

The released benchmark covers **51 tasks** from DSBench's original tabular
slice.

- Prompts + per-task evaluators: <https://huggingface.co/datasets/anonymous222bit/Ambig-DS-T>
- Raw CSV data: not redistributed (Kaggle competition rules) — fetch from
  Kaggle, then rebuild the ambig arm locally using the recipe in
  `_manifest.json`.

---

## Quickstart: evaluate a model against the benchmark

To **evaluate** an agent against the released benchmark (rather than rebuild
it), use the sibling evaluator at
[`evaluate/ambig_ds_target/`](../../evaluate/ambig_ds_target/) — it pulls
the prompts/evaluators from HF, regenerates the ambig CSVs locally from the
recorded seeds, and shells each task out to `release/tasks/<slug>/eval.py`
(the standard DSBench CLI: `--answer_file --predict_file --path --name`).

---

## Reproducing the benchmark from scratch

The full builder pipeline lives in [`pipeline_DSBench/`](pipeline_DSBench/).
It converts the upstream **DSBench tabular slice** (pre-split
`data_resplit/<slug>/`) into the target-ambig variant and implements the
paper's per-task bisected noise calibration.

For an end-to-end walkthrough (workspace layout, env vars, single-task
quickstart on `bike-sharing-demand`, multi-task scaling) see
[pipeline_DSBench/README.md](pipeline_DSBench/README.md).

Minimal flow:

```bash
export AMBIG_DSBENCH_ROOT=/path/to/workspace      # contains Dataset/, DSBench/
cd pipeline_DSBench
python step_1_generate_decoy.py        --tasks_csv ... --src_data_root ... --out_root ...
python step_2_generate_ambig_prompts.py --slug <slug> --env-file /path/to/.env
python step_2b_llm_verify.py            --tasks_csv ... --run --env-file /path/to/.env  # optional
python step_4_build_release.py          --out ./release --tasks <slug>
python step_5_upload_to_hf.py           --skip-build --out ./release   # optional
```

---

## Repository layout

```
create_datasets/ambig_ds_target/
├── README.md                          # this file
└── pipeline_DSBench/                  # the only sub-pipeline (see its README)
    ├── _llm_client.py                 # shared OpenAI-compatible client
    ├── step_1_generate_decoy.py       # rank-map + bisected noise calibration
    ├── step_2_generate_ambig_prompts.py  # LLM rewrite: task.txt -> task_ambig.txt
    ├── step_2b_llm_verify.py              # LLM verification of ambig prompts
    ├── step_3_audit.py                # release audit (5 invariants per task)
    ├── step_4_build_release.py        # assemble HF-shaped release/ tree
    ├── step_5_upload_to_hf.py         # push release/ to HuggingFace
    └── validate_noise_ablation.py     # paper Table-6 noise sweep
```

---

## Citation

If you use this benchmark, please cite the paper (link forthcoming).
