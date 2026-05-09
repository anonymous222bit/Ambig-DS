"""Answerer LLM for the one-turn clarification protocol (target ambiguity).

Used by step_5_run_agent_clarify.py. Given the agent's clarifying question and
the per-task decoy manifest, returns a short, honest answer in <=2 sentences.

In the target-ambig variant, the agent has been given a prompt that:
  - hides the original target column name,
  - exposes two candidate target columns `val_1` and `val_2` in train.csv,
  - leaves it to the agent to figure out which one is the real target.

Allowed disclosures:
  - which of `val_1` / `val_2` is the real target,
  - whether the task is classification vs. regression (already in prompt),
  - what the original target column was conceptually (very short, e.g.
    "the binary churn label", "the daily store sales count") — only when
    the agent asks specifically about target semantics.

Forbidden:
  - test labels, test stats, model recommendations, code, validation advice,
  - feature-set/encoding decisions, ensembling, hyperparameters,
  - confirming / denying the agent's *prediction values* for any specific row.

Configuration:
    OPENAI_API_KEY   required
    OPENAI_BASE_URL  optional
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI


TARGET_SYSTEM_PROMPT = """\
You are an expert helping a data-science agent on a Kaggle-style task.

The agent gets ONE clarifying question. The task description it received was
mutated to HIDE the true target column. Two candidate columns are present in
train.csv (`val_1` and `val_2`); exactly one is the real target, the other
is a calibrated decoy with a similar marginal distribution. The agent does
not know which is which.

You will be told:
  - the task name,
  - which candidate column (`val_1` or `val_2`) is the real target,
  - the original target column's name (do NOT volunteer this unless the
    agent specifically asks for the target's *meaning*),
  - the task type (classification / regression),
  - the agent's question.

Reply in 1-2 short sentences. Be direct and honest.

CRITICAL — scope rule:
  Answer ONLY what the agent literally asked. Do NOT volunteer extra
  information.

Allowed (only when the question is directly about it):
  - Confirm which candidate column is the real target (by name: `val_1`
    or `val_2`).
  - State the task type (classification or regression) — though this is
    typically already in the prompt.
  - When the agent asks about target *semantics*, give a short conceptual
    description (1 short clause) of what the column represents. Keep it
    conceptual; do not paste the exact original column name unless the
    agent has already named it correctly.
  - Confirm or correct an explicit assumption the agent stated.

Forbidden:
  - Do not reveal any test-set labels, statistics, or per-row truth values.
  - Do not suggest models, hyperparameters, validation strategies,
    cross-validation folds, or feature engineering.
  - Do not write code.
  - Do not name the decoy column or describe how it was constructed.
  - Do not advise on submission format details beyond what the prompt says.
  - Do not mention internal benchmark details (manifest files, diagnostics,
    decoy seeds, etc.).

If the agent asks you to do the modelling, write code, choose a toolchain,
suggest hyperparameters, choose features/encodings, or anything outside the
target-identity / target-semantics scope, reply with EXACTLY this single
line and nothing else:
"REFUSE: that's outside the scope of one clarifying answer."

If the agent's question is empty or just "NONE", reply: "NONE".
"""


FULL_SYSTEM_PROMPT = """\
You are an expert helping a data-science agent on a Kaggle-style task.

The agent received the ORIGINAL, unmodified task description: the true
target column is named in the task prompt and in `sample_submission.csv`,
and `train.csv` exposes that column directly (there is NO `val_1`/`val_2`
renaming and NO decoy column). There is no target ambiguity in this
setting; the agent gets one clarifying question regardless.

You will be told:
  - the task name,
  - the real target column name (as it appears in train.csv and the
    sample submission),
  - the task type (classification / regression),
  - the agent's question.

Reply in 1-2 short sentences. Be direct and honest.

CRITICAL — scope rule:
  Answer ONLY what the agent literally asked. Do NOT volunteer extra
  information.

Allowed (only when the question is directly about it):
  - Confirm the real target column name and a short conceptual description
    of what it represents.
  - State the task type (classification or regression) — though this is
    typically already in the prompt.
  - Confirm or correct an explicit assumption the agent stated about the
    target's identity or semantics.

Forbidden:
  - Do NOT mention `val_1` or `val_2` or any decoy / ambiguity construction
    — none of that exists in this variant.
  - Do not reveal any test-set labels, statistics, or per-row truth values.
  - Do not suggest models, hyperparameters, validation strategies,
    cross-validation folds, or feature engineering.
  - Do not write code.
  - Do not advise on submission format details beyond what the prompt says.
  - Do not mention internal benchmark details (manifest files, diagnostics,
    decoy seeds, etc.).

If the agent asks you to do the modelling, write code, choose a toolchain,
suggest hyperparameters, choose features/encodings, or anything outside the
target-identity / target-semantics scope, reply with EXACTLY this single
line and nothing else:
"REFUSE: that's outside the scope of one clarifying answer."

If the agent's question is empty or just "NONE", reply: "NONE".
"""


def _build_user_message(*, question: str, task_name: str,
                        true_target_column: str, original_target_name: str,
                        target_type: str, variant: str) -> str:
    if variant == "full":
        lines = [
            f"Task: {task_name}",
            f"Real target column (in train.csv and sample_submission.csv): {original_target_name}",
            f"Task type: {target_type}",
            "",
            "Agent's clarifying question:",
            question.strip() or "(empty)",
        ]
    else:
        lines = [
            f"Task: {task_name}",
            f"Real target column in train.csv: {true_target_column}",
            f"(Original column name, for context only): {original_target_name}",
            f"Task type: {target_type}",
            "",
            "Agent's clarifying question:",
            question.strip() or "(empty)",
        ]
    return "\n".join(lines)


def answer_clarify_target(*, question: str, task_name: str,
                          true_target_column: str, original_target_name: str,
                          target_type: str, model: str, api_key: str,
                          base_url: str, variant: str = "ambig_target",
                          max_tokens: int = 1200,
                          temperature: float = 0.2) -> dict[str, Any]:
    """Call the target-aware answerer LLM. Returns {answer, raw, refused}.

    ``variant`` selects the system prompt:
      - "ambig_target": uses TARGET_SYSTEM_PROMPT (val_1/val_2 disclosure).
      - "full": uses FULL_SYSTEM_PROMPT (no decoy/ambig context).

    Note on max_tokens: 1200 is high enough to accommodate reasoning models
    (e.g. gemini_3_flash, o-series) that consume hundreds of hidden reasoning
    tokens before producing visible text. Older non-reasoning models will use
    only the ~30-50 tokens needed for a 1-2 sentence reply.
    """
    if not question or question.strip().upper() == "NONE":
        return {"answer": "NONE", "refused": False, "raw": ""}

    user_msg = _build_user_message(
        question=question, task_name=task_name,
        true_target_column=true_target_column,
        original_target_name=original_target_name,
        target_type=target_type,
        variant=variant,
    )
    system_prompt = FULL_SYSTEM_PROMPT if variant == "full" else TARGET_SYSTEM_PROMPT
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    if not text and finish_reason == "length":
        text = (
            "(answerer hit max_tokens before producing visible text — "
            "likely a reasoning model with too small a budget; "
            "raise clarify_answerer.answer_clarify_target(max_tokens=...).)"
        )
    refused = text.startswith("REFUSE:")
    return {"answer": text, "refused": refused, "raw": user_msg, "finish_reason": finish_reason}
