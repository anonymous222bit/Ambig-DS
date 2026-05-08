# DSBench — Ambiguity Benchmarks for ML-Engineering Agents

This repository contains two benchmarks for evaluating coding agents on
data-science tasks under **prompt ambiguity**, plus the pipelines that
build them and the evaluators that run agents against them.

| Benchmark | Ambiguity type | What's hidden | Tasks | HuggingFace |
|---|---|---|---|---|
| **Ambig-DS-M** | *Metric* ambiguity | Evaluation metric is redacted from the prompt | 61 Kaggle competitions | [`anonymous222bit/Ambig-DS-M`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-M) |
| **Ambig-DS-T** | *Target* ambiguity | A statistically plausible **decoy column** is injected so the true label column is unclear | 51 tabular tasks (from upstream DSBench) | [`anonymous222bit/Ambig-DS-T`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-T) |

Each task ships in two arms: `full` (original prompt) and
`ambig_metric` / `ambig_target` (rewritten with the relevant signal removed).

---

## Repository layout

```
DSBench/
├── create_datasets/              # build the HF datasets
│   ├── ambig_ds_metric/pipeline/                  # 3-step metric pipeline
│   └── ambig_ds_target/pipeline_DSBench/          # 5-step target pipeline
└── evaluate/                     # run agents + grade submissions
    ├── ambig_ds_metric/                           # 4-step metric evaluator (uses MLE-bench)
    └── ambig_ds_target/                           # 5-step target evaluator (uses per-task DSBench eval.py)
```

Each subfolder has its own `README.md` with detailed steps and flags:

- [create_datasets/ambig_ds_metric/README.md](create_datasets/ambig_ds_metric/README.md)
- [create_datasets/ambig_ds_target/README.md](create_datasets/ambig_ds_target/README.md)
- [create_datasets/ambig_ds_target/pipeline_DSBench/README.md](create_datasets/ambig_ds_target/pipeline_DSBench/README.md)
- [evaluate/ambig_ds_metric/README.md](evaluate/ambig_ds_metric/README.md)
- [evaluate/ambig_ds_target/README.md](evaluate/ambig_ds_target/README.md)

---

## Quickstart — evaluate an agent

Most users want to **evaluate** an agent on an existing benchmark, not
rebuild it. Pick the track and follow its evaluator README.

> **macOS first-time setup:** mle-bench needs **Python ≥ 3.11**, opencode
> needs a per-user `npm` prefix, kaggle needs `~/.kaggle/kaggle.json`,
> and corporate TLS proxies (Zscaler etc.) need their root CA appended
> to certifi. The metric evaluator's
> [Verified setup section](evaluate/ambig_ds_metric/README.md#verified-setup)
> walks through every step end-to-end. Highlights below.

### 1. Python 3.12 venv

```bash
# macOS ships Python 3.9; install a newer one.
# Option A: uv  (recommended when you have unrestricted network access)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv

# Option B: standalone build (works behind corporate proxies)
curl -L -o /tmp/python.tar.gz \
  'https://github.com/astral-sh/python-build-standalone/releases/download/20260504/cpython-3.12.13+20260504-aarch64-apple-darwin-install_only_stripped.tar.gz'
mkdir -p ~/.local/python312 && tar xzf /tmp/python.tar.gz -C ~/.local/python312
~/.local/python312/python/bin/python3 -m venv .venv

source .venv/bin/activate
pip install --upgrade pip
pip install 'openai>=1.0' 'huggingface_hub>=0.24' 'pandas>=2.0' py7zr \
            numpy scikit-learn
pip install -e 'git+https://github.com/openai/mle-bench.git#egg=mlebench'   # metric track only
```

### 2. Kaggle credentials (metric track)

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{"username":"<your-kaggle-username>","key":"<your-kaggle-key>"}
JSON
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions list | head -3        # sanity check
```

Get `<username>` and `<key>` from kaggle.com → Account → "Create New API
Token". Then **accept each competition's rules in a browser** (e.g.
<https://www.kaggle.com/c/spooky-author-identification/rules>) before
running `mlebench prepare`, otherwise it returns 403.

### 3. Corporate TLS proxy (Zscaler)

If `kaggle competitions list` returns `[SSL: CERTIFICATE_VERIFY_FAILED]`,
your traffic is being intercepted. Append the proxy's root CA to
certifi's bundle:

```bash
security find-certificate -a -p -c "Zscaler" /Library/Keychains/System.keychain \
  >> "$(python -c 'import certifi; print(certifi.where())')"
```

### 4. opencode CLI

```bash
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH        # add to ~/.zshrc to persist
npm install -g opencode-ai
opencode --version
```

(Or use the internal `claw` CLI, which is also supported via `--agent claw`.)

### 5. Project `.env` and run the pipelines

```bash
cp .env.example .env && $EDITOR .env
set -a && source .env && set +a

# Metric track
cd evaluate/ambig_ds_metric
python step_1_setup_benchmark.py --benchmark-dir ./benchmark \
    --tasks spooky-author-identification
python step_2_run_agent.py --benchmark-dir ./benchmark \
    --variant ambig_metric --model <model-id> \
    --tasks spooky-author-identification \
    --agent opencode --agent-bin "$HOME/.npm-global/bin/opencode"

# Target track
cd ../ambig_ds_target
python step_1_setup_benchmark.py --benchmark-dir ./benchmark \
    --release-source hf --hf-repo anonymous222bit/Ambig-DS-T
python step_4_run_agent.py --benchmark-dir ./benchmark \
    --variant ambig_target --model <model-id>
```

> **Heads-up — `mle-bench` leaderboards are Git LFS.** Step 2's grader
> needs them; if `git lfs` was missing when you `pip install -e`'d
> mle-bench you'll see `AssertionError: Leaderboard must have a 'score'
> column.`. Fix per the
> [Step 1.5 helper](evaluate/ambig_ds_metric/README.md#step-15--fetch-mle-bench-leaderboards-git-lfs)
> in the metric README (one `curl` per slug; no auth required).

---

## Configuration — `.env`

All scripts read credentials and paths from environment variables. The
recommended workflow is to keep them in a `.env` file at the repo root and
source it before running anything.

### 1. Create the file

A template is provided at [`.env.example`](.env.example). Copy it and fill
in the blanks:

```bash
cp .env.example .env
$EDITOR .env             # fill in API keys, paths, etc.
```

### 2. Load it into your shell

```bash
set -a && source .env && set +a
```

(`set -a` makes every assignment in the sourced file an exported variable.)

### 3. What to fill in

| Variable | Required for | What to put |
|---|---|---|
| `OPENAI_API_KEY` | every LLM call (creators + evaluators + agents) | API key for any OpenAI-compatible chat endpoint |
| `OPENAI_BASE_URL` | non-default endpoints | e.g. `https://api.openai.com/v1`, an Azure/vLLM gateway, etc. Omit to use OpenAI's default. |
| `AMBIG_LLM_MODEL` | creator pipelines (LLM rewrites) | Default chat model id (e.g. `gpt-4o-mini`, `anthropic_claude_opus_4_7`) |
| `HF_TOKEN` | `step_3_upload_to_hf.py` / `step_5_upload_to_hf.py` | HuggingFace write token (`hf_...`) |

The metric evaluator additionally needs **Kaggle credentials** at
`~/.kaggle/kaggle.json` (chmod 600), and you must accept each
competition's rules on kaggle.com before `mlebench prepare` will download
its data.

---

## Citation

```bibtex
@article{ambig-ds-2026,
  title = {Measuring the Impact of Prompt Ambiguity on ML-Engineering Agents},
  year  = {2026}
}
```

## License & attribution

This repository contains two layers, licensed separately:

- **Original code we wrote** (creator pipelines, evaluators' wrappers,
  agent runners, oracle, scoring glue): **MIT** — see [`LICENSE`](LICENSE).
- **Released datasets on Hugging Face**
  ([`Ambig-DS-M`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-M),
  [`Ambig-DS-T`](https://huggingface.co/datasets/anonymous222bit/Ambig-DS-T)) —
  the *novel additions* (ambiguous prompt rewrites, manifests, scoring
  metadata) are released under **CC-BY-NC-4.0**, propagating the DSBench
  non-commercial dataset restriction and conservatively respecting
  per-competition Kaggle rules. Each dataset declares its license in its
  YAML frontmatter and Croissant file.

This repository **vendors and depends on** third-party code and data:

- [DSBench](https://github.com/LiqiangJing/DSBench) (MIT, code only;
  data: non-commercial research/education) — basis for Ambig-DS-T,
  including the per-task `eval.py` files we redistribute inside the
  `Ambig-DS-T` dataset.
- [MLE-bench](https://github.com/openai/mle-bench) (MIT, code only) —
  basis for Ambig-DS-M; we depend on `mlebench` as a Python package and
  reference its per-competition `description.md` and graders.
- The underlying **Kaggle competition data** (training data, test data,
  labels, media) — © each competition's host, governed by each
  competition's own terms. **We do not redistribute these.** Users
  download them via the Kaggle CLI and the upstream `mlebench prepare`
  / DSBench data-preparation pipelines.

Full upstream copyright and license notices are reproduced in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), as required by
the MIT license. If you redistribute any portion of this repository or
either dataset, you must preserve those notices.
