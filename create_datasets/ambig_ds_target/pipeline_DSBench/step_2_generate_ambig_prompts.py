"""Step 4: rewrite each task's normal prompt into a target-ambiguous variant.

For DSBench tasks the upstream DSBench repo already ships a clean per-task
prompt at::

    $AMBIG_DSBENCH_ROOT/Dataset/data_modeling/data/data/task/<slug>.txt

This script uses that file directly as the "normal" prompt, then runs the
LLM rewrite that produces the target-ambiguous variant.

Reads:
  $AMBIG_DSBENCH_ROOT/Dataset/data_modeling/data/data/task/<slug>.txt
  $AMBIG_DSBENCH_ROOT/final_data_v3/target_ambig/data/<slug>/_manifest.json
    (written by step_1_generate_decoy.py)

Writes (both — the canonical location is the second one):
  $AMBIG_DSBENCH_ROOT/final_data_v3/target_ambig/data/<slug>/ambig_prompt.txt
  $AMBIG_DSBENCH_ROOT/final_data_v3/target_ambig/data_modeling/data/data/task_ambig_target_v3_gen/<slug>.txt

Credentials:
  Reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment first, then
  falls back to a .env file. Pass --env-file to point at a specific .env. --env-file values
  always override existing environment variables.

Usage:
  # all 6 dsbench-remainder tasks from the spec CSV
  python step_4_generate_ambig_prompts.py \
      --tasks_csv $AMBIG_DSBENCH_ROOT/final_data_v3/target_ambig/dsbench_remainder_tasks.csv \
      --env-file  /path/to/.env

  # single slug
  python step_4_generate_ambig_prompts.py --slug santander-customer-transaction-prediction
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from _llm_client import make_client, call_llm, DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Prompt + validator. Kept in lock-step with the sibling Kaggle-scrape
# pipeline (../pipeline/make_ambig_prompt.py); update both if you change
# either.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You convert a normal Kaggle-style task description into a TARGET-AMBIGUOUS
variant for an autonomous data-science benchmark.

IMPORTANT — what this benchmark measures:
We test whether an agent SPONTANEOUSLY DETECTS that the task specification is
under-determined (two plausible target columns exist in train.csv) and runs a
diagnostic to choose the correct one. The prompt MUST therefore READ AS A
NORMAL, SINGLE-TARGET TASK. Do NOT hint, mention, or describe the ambiguity.
A strong agent has to notice it on its own by inspecting the data.

HARD RULES:
1. The training file ships two candidate target columns named `val_1` and
   `val_2` (this is a fact about the data on disk). DO NOT mention `val_1`,
   `val_2`, "two candidate columns", "decoy", "ambiguous target",
   "identify the target", "similar marginal", or any synonym anywhere in
   the output. The prompt must not signal that target ambiguity exists.
2. Replace every mention of the original target column name (and any obvious
   synonym for the predicted concept) with neutral wording. Do NOT name the
   concept anywhere in the output.
3. Replace every original feature column name with its anonymized counterpart
   from the provided feature_map. Do this everywhere it appears, including
   prose, tables, value-enumerations, and units. Strip semantic hints about
   feature meaning (e.g. "temperature in Celsius", "day of the week",
   "binary feature", "ordinal") when those hints would let a reader
   reverse-engineer the task from the original Kaggle competition.
4. In the Submission File section:
   - Keep the metric and per-row format intact.
   - Use a generic header column name `prediction` instead of the real target.
   - Update the example block accordingly.
5. Remove `sample_submission.csv` from the Files list.
6. In the Files / Columns / Data Fields section: list the columns that exist
   in train.csv, including `val_1` and `val_2`, but describe them with the
   SAME neutral one-liner each (e.g. "numeric column" / "label column")
   so neither stands out. Do not annotate them as "candidate", "target",
   or "decoy". If the input prompt did not list per-column descriptions,
   do not invent one — just list column names.
7. NUMERIC FACTS: The ONLY row counts, feature counts, column counts, and
   per-file shapes you may state in the output are the ones explicitly
   listed in the "Decoy manifest" block of the user message (n_train,
   n_test, n_features). Do NOT copy any number from the input "Normal
   task prompt" (those numbers refer to the original Kaggle dataset, not
   the resplit shipped with this benchmark, and are almost always wrong
   here). Do NOT recall numbers from memory of the original Kaggle
   competition. Do NOT invent or estimate. If a number is not in the
   manifest, OMIT it entirely (rephrase the sentence to drop the count,
   e.g. "the training set" instead of "the 480,000-row training set").
8. Keep dataset narrative and evaluation metric description faithful to the
   input, but strip phrases that uniquely identify the original Kaggle
   competition (acknowledgements naming the dataset, citation titles
   containing the target concept, narrative giving away units of the
   target, etc.).
9. Use the same section headers as the input, in the same order.
10. Do NOT use markdown fences. Output plain text only.
"""

USER_TEMPLATE = """\
# Decoy manifest
- task: {task}
- true_target_column (hidden from agent): {true_target}
- decoy_column (hidden from agent): {decoy}
- original_target_name (must NOT appear in your output): {original_target}
- target_type: {target_type}
- id_column: {id_column}
- n_features: {n_features}
- n_train: {n_train}
- n_test: {n_test}

# Feature renaming map (apply EVERY occurrence)
{feature_map_block}

# Anonymized feature columns (in order)
{anon_features}

# Normal task prompt (input)
<<<NORMAL
{normal_prompt}
NORMAL>>>

Produce the target-ambiguous variant now.
"""


def build_user_prompt(normal: str, manifest: dict) -> str:
    fmap = manifest.get("feature_map", {})
    feature_map_block = "\n".join(f"- {orig!r}  ->  {anon}"
                                  for orig, anon in fmap.items()) or "(empty)"
    return USER_TEMPLATE.format(
        task=manifest.get("task", "?"),
        true_target=manifest.get("true_target_column", "val_1"),
        decoy=manifest.get("decoy_column", "val_2"),
        original_target=manifest.get("original_target_name", "?"),
        target_type=manifest.get("target_type", "?"),
        id_column=manifest.get("id_column", "id"),
        n_features=manifest.get("n_features", "?"),
        n_train=manifest.get("n_train", "?"),
        n_test=manifest.get("n_test", "?"),
        feature_map_block=feature_map_block,
        anon_features=", ".join(manifest.get("anon_feature_columns", [])) or "(empty)",
        normal_prompt=normal.strip(),
    )


# Phrases that would tip the agent off that target ambiguity exists.
# The whole point of the benchmark is that the agent must NOTICE the
# ambiguity on its own; any of these strings in the output is a leak.
_AMBIG_DISCLOSURE_PHRASES = (
    "candidate target", "candidate label", "two candidate",
    "two target", "two label", "decoy", "ambiguous",
    "identify the target", "identify which", "which column",
    "similar marginal", "true target", "real target",
    "is the target", "genuine target",
)


def validate_ambig(text: str, manifest: dict) -> list[str]:
    """Cheap leak checks. Returns a list of warnings (empty if clean)."""
    import re as _re
    warnings = []
    low = text.lower()
    orig = manifest.get("original_target_name", "")
    if orig and len(orig) >= 3:
        # Whole-word match so e.g. "id" doesn't match "identifier".
        if _re.search(rf"(?<![A-Za-z0-9_]){_re.escape(orig)}(?![A-Za-z0-9_])",
                      text, _re.IGNORECASE):
            warnings.append(f"original target name '{orig}' still appears")
    for orig_col in manifest.get("feature_map", {}).keys():
        # only flag multi-character column names to avoid false positives on 'id'
        if len(orig_col) < 3:
            continue
        # Skip already-anonymized originals (e.g. f_00 -> f_01 renames). The
        # model legitimately needs to use f_NN tokens as the NEW names, so
        # flagging an old f_NN as a "leak" creates an unfixable contradiction
        # and they aren't semantic leaks anyway.
        if _re.fullmatch(r"f_?\d+", orig_col):
            continue
        if _re.search(rf"(?<![A-Za-z0-9_]){_re.escape(orig_col)}(?![A-Za-z0-9_])",
                      text):
            warnings.append(f"original feature column '{orig_col}' still appears")
    if "sample_submission" in text:
        warnings.append("sample_submission.csv still mentioned")
    # Acknowledgements / citation paragraphs typically name the source
    # competition, dataset, or unique provenance and let the reader
    # reverse-engineer the original Kaggle task. Hard rule #8 forbids them.
    if _re.search(r"(?im)^\s*Acknowledg", text):
        warnings.append("Acknowledgements section present (identifies source)")
    if _re.search(r"(?im)\bcite\b|\bcitation\b", text):
        warnings.append("Citation phrasing present (identifies source)")
    # Reverse of the old check: val_1/val_2 must NOT be flagged in prose.
    # (They will appear in the on-disk CSV but the prompt must not announce
    # them — the agent has to discover them by listing columns.)
    for phrase in _AMBIG_DISCLOSURE_PHRASES:
        if phrase in low:
            warnings.append(f"ambiguity disclosure leak: '{phrase}' appears in prompt")
    return warnings


def _build_retry_message(warnings: list[str]) -> str:
    """Construct an explicit follow-up user message that lists every leak the
    validator detected and instructs the model to regenerate the prompt
    obeying the original system rules. Strings are quoted so the model can
    grep its previous output for them."""
    bullet_lines = []
    for w in warnings:
        bullet_lines.append(f"- {w}")
    bullets = "\n".join(bullet_lines)
    return (
        "Your previous output VIOLATED the hard rules. The following leaks "
        "were detected and MUST be removed (do not paraphrase, do not move, "
        "REMOVE):\n\n"
        f"{bullets}\n\n"
        "Regenerate the ENTIRE target-ambiguous prompt from scratch, this "
        "time obeying every rule in the system message. In particular: "
        "(a) every original feature column name listed above must be "
        "replaced with its f_NN counterpart from the feature map; "
        "(b) the original target concept name must not appear anywhere, "
        "even in narrative or evaluation prose -- rephrase to a neutral "
        "noun like 'the outcome' or 'the value'; "
        "(c) drop any Acknowledgements / Citation / dataset-attribution "
        "paragraph that names the source competition or dataset; "
        "(d) keep the same section headers and the same evaluation metric. "
        "Output plain text only, no markdown fences, no commentary."
    )


def load_env_file(path: Path) -> None:
    """Minimal dotenv: load OPENAI_*/AMBIG_* vars from a KEY=VAL file.

    Always overrides existing process env for these keys -- the user passed
    --env-file precisely because they want this file to win over whatever
    stale OPENAI_API_KEY / OPENAI_BASE_URL is hanging around in the shell.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key.startswith(("OPENAI_", "AMBIG_")):
            os.environ[key] = val


def rewrite_one(slug: str, *, root: Path, model: str, force: bool,
                client, max_retries: int = 4) -> int:
    normal_path = root / "Dataset" / "data_modeling" / "data" / "data" / "task" / f"{slug}.txt"
    manifest_path = root / "final_data_v3" / "target_ambig" / "data" / slug / "_manifest.json"
    out_local = root / "final_data_v3" / "target_ambig" / "data" / slug / "ambig_prompt.txt"
    out_canon = (root / "final_data_v3" / "target_ambig" / "data_modeling"
                 / "data" / "data" / "task_ambig_target_v3_gen" / f"{slug}.txt")
    rejected_dir = (root / "final_data_v3" / "target_ambig"
                    / "data_modeling" / "data" / "data" / "_rejected")

    if not normal_path.exists():
        print(f"[{slug}] SKIP — missing normal prompt: {normal_path}")
        return 2
    if not manifest_path.exists():
        print(f"[{slug}] SKIP — missing manifest: {manifest_path} (run step_3b first)")
        return 2
    if out_canon.exists() and not force:
        print(f"[{slug}] SKIP — already exists: {out_canon} (use --force)")
        return 0

    normal = normal_path.read_text()
    manifest = json.loads(manifest_path.read_text())
    base_user = build_user_prompt(normal, manifest)

    text = ""
    warnings: list[str] = []
    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            user_msg = base_user
            print(f"[{slug}] calling {model} (attempt {attempt}/{max_retries}) ...",
                  flush=True)
        else:
            # Append the previous attempt + an explicit list of leaks so the
            # model can see exactly what to remove. Keep the original
            # instructions in scope by prepending the original user message.
            user_msg = (
                base_user
                + "\n\n# Previous attempt (REJECTED — do NOT repeat its leaks)\n"
                + "<<<PREVIOUS\n"
                + text.rstrip()
                + "\nPREVIOUS>>>\n\n"
                + _build_retry_message(warnings)
            )
            print(f"[{slug}] retrying after {len(warnings)} leak(s) "
                  f"(attempt {attempt}/{max_retries}) ...", flush=True)

        text = call_llm(client, SYSTEM_PROMPT, user_msg, model=model)
        if not text.strip():
            print(f"[{slug}]   empty response on attempt {attempt}")
            warnings = ["empty LLM response"]
            continue
        text = text.strip() + "\n"

        warnings = validate_ambig(text, manifest)
        if not warnings:
            print(f"[{slug}]   clean on attempt {attempt}")
            break
        for w in warnings:
            print(f"  WARN (attempt {attempt}): {w}")

    if warnings:
        # All retries exhausted with leaks remaining. Park the last attempt
        # under _rejected/ so it can be inspected / patched manually.
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rej_path = rejected_dir / f"{slug}.txt"
        rej_path.write_text(text)
        rej_log = rejected_dir / f"{slug}.warnings.txt"
        rej_log.write_text("\n".join(warnings) + "\n")
        print(f"[{slug}] REJECTED after {max_retries} attempts; "
              f"wrote {rej_path}")
        return 1

    out_local.parent.mkdir(parents=True, exist_ok=True)
    out_canon.parent.mkdir(parents=True, exist_ok=True)
    out_local.write_text(text)
    out_canon.write_text(text)
    print(f"[{slug}] wrote {len(text)} chars; clean")
    print(f"  -> {out_local}")
    print(f"  -> {out_canon}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", help="Single task slug.")
    g.add_argument("--tasks_csv", help="CSV with a 'task' column listing slugs.")
    ap.add_argument("--root", default=os.environ.get("AMBIG_DSBENCH_ROOT", ""),
                    help="Workspace root (default: $AMBIG_DSBENCH_ROOT).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--env-file", default=None,
                    help="Optional .env file to load OPENAI_* vars from.")
    args = ap.parse_args()

    if args.env_file:
        load_env_file(Path(args.env_file))

    if not args.root:
        sys.exit("Set --root or AMBIG_DSBENCH_ROOT.")
    root = Path(args.root).resolve()

    if args.slug:
        slugs = [args.slug]
    else:
        df = pd.read_csv(args.tasks_csv)
        if "task" not in df.columns:
            sys.exit(f"--tasks_csv must have a 'task' column (got {list(df.columns)})")
        slugs = df["task"].astype(str).tolist()

    # Read AFTER load_env_file so --env-file values win.
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("AMBIG_LLM_MODEL", args.model)
    print(f"Using base_url={base_url}  model={model}")
    client = make_client(base_url=base_url)
    bad = 0
    for slug in slugs:
        rc = rewrite_one(slug, root=root, model=model,
                         force=args.force, client=client)
        if rc not in (0,):
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
