"""Answerer LLM for the one-turn clarification protocol (metric ambiguity).

Used by step_3_run_agent_clarify.py. Given the agent's clarifying question and the
ground-truth metric manifest entry, returns a short, honest answer in <=2 sentences.

Configuration:
    OPENAI_API_KEY   required
    OPENAI_BASE_URL  optional (defaults to https://api.openai.com/v1)
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


METRIC_SYSTEM_PROMPT = """You are an expert helping a data-science agent on a Kaggle-style task.

The agent gets ONE clarifying question. The task description it received had
its evaluation/metric section neutralised half of the time — the agent may NOT know the exact
scoring metric, scoring direction, or required submission value type
(probability vs. hard label, top-K, etc.).

You will be told:
  - the task name,
  - the true evaluation metric, a short description, and the required
    submission format,
  - whether the metric is lower-better or higher-better,
  - the agent's question
  - whether this is a fully specified task or one without metric specification.

Reply in 1-2 short sentences. Be direct and honest.

CRITICAL — scope rule:
  Answer ONLY what the agent literally asked, in the metric / scoring /
  submission-format domain. Do NOT volunteer extra information.

Allowed (only when the question is directly about it):
  - The exact name and a brief description of the evaluation metric.
  - Whether higher or lower scores are better.
  - The required submission column structure and value type
    (probability vs. hard label, score per class, top-K list, continuous
    regression value, etc.).
  - Any metric-specific quirks (e.g. K=5 for MAP@K; only certain target
    columns are scored; scoring is on inspiratory phase only; eps clipping;
    asymmetry of RMSLE).
  - Confirming or correcting an explicit assumption the agent stated.

Forbidden:
  - Do not reveal any test-set labels or test-set statistics.
  - Do not give exact CV scores, thresholds, or recommend models / toolchains
    / hyperparameters / training procedures / validation strategies.
  - Do not write code for the agent.
  - Do not copy the task description verbatim.
  - Do not mention internal benchmark details (manifest files, ground-truth
    paths, etc.).
  - Do not advise on feature-set decisions, feature encoding, train/val split,
    cross-validation, ensembling, or any modelling workflow detail.
  - Do not state or confirm rules about external data / pre-trained models.

If the agent asks you to do the modelling, write code, choose a toolchain,
suggest hyperparameters, choose features/encodings, confirm external-data
rules, or anything outside the metric / scoring / submission-format scope,
reply with EXACTLY this single line and nothing else:
"REFUSE: that's outside the scope of one clarifying answer."

If the agent's question contains BOTH an off-topic part AND an explicit
in-scope part, answer ONLY the in-scope part in 1-2 sentences.

If the agent's question is empty or just "NONE", reply: "NONE".
"""


VARIANT_ADDENDUM = {
    "ambig_metric": (
        "\n\nVARIANT NOTE: the agent's evaluation/metric section WAS\n"
        "neutralised. In-scope metric/scoring/submission-format questions\n"
        "are legitimate; answer them within the rules above.\n"
    ),
    "full": (
        "\n\nVARIANT NOTE: the agent ALREADY HAS the full evaluation/metric\n"
        "section in its task prompt. Therefore:\n"
        "  - Do NOT volunteer the metric name, direction, submission format,\n"
        "    or any metric quirks. Only CONFIRM or CORRECT an explicit,\n"
        "    specific assumption the agent stated in its question.\n"
        "  - If the question is a generic workflow check rather than a specific\n"
        "    assumption to confirm, refuse with the standard REFUSE line.\n"
        "  - Be especially strict about the modelling/feature/encoding/\n"
        "    external-data forbidden list above.\n"
    ),
}


def _build_metric_user_message(
    *,
    question: str,
    task_name: str,
    metric_name: str,
    metric_description: str,
    submission_format: str,
    is_lower_better: bool,
    notes: str | None,
) -> str:
    direction = "LOWER is better" if is_lower_better else "HIGHER is better"
    lines = [
        f"Task: {task_name}",
        f"True evaluation metric: {metric_name}  ({direction})",
        f"Metric description: {metric_description}",
        f"Required submission format: {submission_format}",
    ]
    if notes:
        lines.append(f"Additional notes: {notes}")
    lines.append("")
    lines.append("Agent's clarifying question:")
    lines.append(question.strip() or "(empty)")
    return "\n".join(lines)


def answer_clarify_metric(
    *,
    question: str,
    task_name: str,
    metric_name: str,
    metric_description: str,
    submission_format: str,
    is_lower_better: bool,
    notes: str | None,
    model: str,
    api_key: str,
    base_url: str,
    variant: str = "ambig_metric",
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Call the metric-aware answerer LLM. Returns {answer, raw, refused}.

    Note on max_tokens: 1200 is high enough to accommodate reasoning models
    (e.g. gemini_3_flash, o-series) that consume hundreds of hidden reasoning
    tokens before producing visible text. Older non-reasoning models will use
    only the ~30-50 tokens needed for a 1-2 sentence reply.
    """
    if not question or question.strip().upper() == "NONE":
        return {"answer": "NONE", "refused": False, "raw": ""}

    user_msg = _build_metric_user_message(
        question=question, task_name=task_name,
        metric_name=metric_name, metric_description=metric_description,
        submission_format=submission_format,
        is_lower_better=is_lower_better, notes=notes,
    )
    system_prompt = METRIC_SYSTEM_PROMPT + VARIANT_ADDENDUM.get(
        variant, VARIANT_ADDENDUM["ambig_metric"]
    )
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
            "raise clarify_answerer.answer_clarify_metric(max_tokens=...).)"
        )
    refused = text.startswith("REFUSE:")
    return {"answer": text, "refused": refused, "raw": user_msg, "finish_reason": finish_reason}
