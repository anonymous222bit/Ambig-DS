#!/usr/bin/env python3
"""LLM-judge audit: classify what metric/loss the agent actually optimized for.

For each (agent_model, condition, task) cell, reads the trajectory + submission
and asks a JUDGE LLM to classify the agent's optimization target into one of:

    Intended | FormBroken | WrongObjective | Abdicated | Invalid | Other

Writes one cached output per cell:
    results/claw_<MODEL>_<VARIANT>[_clarify]/<slug>/_audit.<judge_model>.json

Skip-existing by default. Re-run with --overwrite to rebuild.

Configuration:
    OPENAI_API_KEY   required
    OPENAI_BASE_URL  optional (defaults to https://api.openai.com/v1)

Usage:
    python step_4_judge_audit.py --benchmark-dir ./benchmark \\
        --judge-model gpt-4o \\
        --agent-models gpt-4o \\
        --conditions ambig_metric \\
        --max-tasks 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from _llm_client import load_api_key
from openai import OpenAI


LABELS = ["Intended", "FormBroken", "WrongObjective", "Abdicated", "Invalid", "Other"]


def load_tasks(benchmark_dir: Path) -> list[str]:
    return [l.strip() for l in (benchmark_dir / "task_list.txt").read_text().splitlines()
            if l.strip()]


def load_manifest(benchmark_dir: Path) -> dict[str, dict]:
    manifest_path = benchmark_dir / "prompts" / "_metric_manifest.json"
    if not manifest_path.exists():
        manifest_path = benchmark_dir / "metric_manifest.json"
    raw = json.loads(manifest_path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ─── trajectory packer ───
def _maybe_json(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def pack_trajectory(traj_path: Path, max_chars: int = 30_000) -> str:
    """Concatenate code blobs and bash commands; head+tail truncate if too long."""
    if not traj_path.exists():
        return "<NO TRAJECTORY FILE>"
    try:
        traj = json.loads(traj_path.read_text())
    except Exception as e:
        return f"<TRAJECTORY UNREADABLE: {e}>"

    summary = (traj.get("summary") or "").strip()
    iters = traj.get("iterations")
    elapsed = traj.get("elapsed_sec")

    parts: list[str] = []
    parts.append(f"AGENT_SUMMARY (self-reported): {summary}")
    parts.append(f"AGENT_RAN: iterations={iters}  elapsed_sec={elapsed}")
    parts.append("")
    parts.append("TOOL_USES (in order):")

    for i, tu in enumerate(traj.get("tool_uses", []) or []):
        name = tu.get("name", "?")
        inp = _maybe_json(tu.get("input", {}))
        if isinstance(inp, dict):
            for key in ("command", "code", "content", "file_text",
                        "new_string", "new_str", "old_string", "old_str", "path"):
                v = inp.get(key)
                if not v or not isinstance(v, str):
                    continue
                parts.append(f"\n--- [{i}] tool={name}  field={key} ---\n{v}")
        else:
            parts.append(f"\n--- [{i}] tool={name} ---\n{inp}")

    text = "\n".join(parts)

    if len(text) <= max_chars:
        return text
    h = (max_chars * 3) // 10
    t = max_chars - h
    return text[:h] + f"\n\n[...truncated {len(text) - max_chars:,} chars...]\n\n" + text[-t:]


def pack_submission_head(csv_path: Path, n_rows: int = 8, max_cols: int = 10) -> str:
    if not csv_path.exists():
        return "<NO SUBMISSION CSV>"
    try:
        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row[:max_cols])
                if i >= n_rows:
                    break
        return "\n".join(",".join(r) for r in rows)
    except Exception as e:
        return f"<SUBMISSION UNREADABLE: {e}>"


# ─── prompt ───
SYSTEM_PROMPT = """You read a data-science agent's full trajectory and final \
submission, and put it into ONE of six buckets describing what the agent did \
with respect to the competition's TRUE metric (given in the manifest).

The agent was given an ambiguous prompt that omits or paraphrases the metric. \
Return STRICT JSON. Quote short evidence verbatim from the trajectory."""


SCHEMA_DOC = """SCHEMA (return exactly this JSON, no preamble, no code fences):

{
  "label":      "Intended" | "FormBroken" | "WrongObjective" | "Abdicated" | "Invalid" | "Other",
  "confidence": <float 0..1>,
  "evidence_quotes": ["<verbatim short snippet>", "..."],
  "rationale":  "<<= 2 sentences>"
}

BUCKETS (pick exactly one):

- "Intended": Agent built a real, non-degenerate model whose optimization
  target is aligned with the TRUE metric (same metric, a positively monotonic
  surrogate, or a hand-coded heuristic that approximates it), AND the
  submission file has the correct form (probabilities for AUC/log-loss; hard
  labels for accuracy/F1; text spans for Jaccard; reals for RMSE), AND the
  predictions actually use the input features (not constant or just the class
  marginals).

- "FormBroken": Agent trained an aligned objective (e.g. cross-entropy / AUC),
  but the final submission form damages the metric. Canonical case: trained
  probabilities, then thresholded to hard 0/1 in the CSV for an AUC or log-loss
  task.

- "WrongObjective": Agent built a real model but optimized something not
  aligned with the TRUE metric (e.g. trained F1 instead of AUC with no
  monotonic relation; minimized when it should have maximized).

- "Abdicated": Submission file exists but is degenerate / not a real model:
  copied a sample submission or baseline, every row predicts the same constant
  or train marginals, trivial dummy with no real modeling step.

- "Invalid": No usable submission. Either no _submission.csv at all, agent
  timed out before writing one, file is corrupt / unreadable / wrong schema.

- "Other": Use ONLY if none of the five buckets above clearly applies.

DECISION ORDER (apply top to bottom, stop at first match):
  1. No submission / unreadable -> Invalid.
  2. Submission is a copied baseline file or every row is the same constant
     -> Abdicated.
  3. Agent trained an aligned objective but submission form contradicts the
     TRUE metric -> FormBroken.
  4. Agent built a real model with a non-aligned optimization target ->
     WrongObjective.
  5. Agent built a real, aligned model with correct submission form -> Intended.
  6. None of the above clearly applies -> Other."""


def build_prompt(*, slug: str, manifest_entry: dict, traj_text: str,
                 submission_head: str, clarify: dict | None) -> tuple[str, str]:
    sys_msg = SYSTEM_PROMPT + "\n\n" + SCHEMA_DOC

    parts = []
    parts.append(f"TASK: {slug}")
    parts.append("")
    parts.append("=== TRUE METRIC (manifest) ===")
    parts.append(f"name:               {manifest_entry.get('metric_name')}")
    parts.append(f"lower_is_better:    {manifest_entry.get('is_lower_better')}")
    parts.append(f"description:        {manifest_entry.get('metric_description')}")
    parts.append(f"submission_format:  {manifest_entry.get('submission_format')}")
    if manifest_entry.get("notes"):
        parts.append(f"notes:              {manifest_entry.get('notes')}")
    parts.append("")
    if clarify is not None:
        parts.append("=== CLARIFY EXCHANGE (if any) ===")
        parts.append(f"asked: {clarify.get('asked')}")
        q = (clarify.get("question") or "").strip()
        a = (clarify.get("answer") or "").strip()
        parts.append(f"question: {q[:500]}")
        parts.append(f"answer:   {a[:500]}")
        parts.append("")
    parts.append("=== AGENT TRAJECTORY ===")
    parts.append(traj_text)
    parts.append("")
    parts.append("=== AGENT SUBMISSION (first rows) ===")
    parts.append(submission_head)
    parts.append("")
    parts.append("Return JSON now.")
    return sys_msg, "\n".join(parts)


# ─── judge call ───
def call_judge(client: OpenAI, judge_model: str, sys_msg: str, user_msg: str,
               max_tokens: int = 1200, retries: int = 2) -> tuple[str, dict]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = (r.choices[0].message.content or "").strip()
            usage = getattr(r, "usage", None)
            usage_d = (
                {"prompt_tokens": usage.prompt_tokens,
                 "completion_tokens": usage.completion_tokens,
                 "total_tokens": usage.total_tokens}
                if usage else {}
            )
            return text, usage_d
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"judge call failed after retries: {last_err}")


def parse_judge_output(text: str) -> tuple[dict | None, str]:
    """Try to extract JSON. Returns (parsed_dict_or_None, raw_or_error)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    try:
        d = json.loads(t)
    except Exception:
        i = t.find("{")
        j = t.rfind("}")
        if i >= 0 and j > i:
            try:
                d = json.loads(t[i:j + 1])
            except Exception as e:
                return None, f"JSON parse failed: {e}; raw={text[:400]}"
        else:
            return None, f"No JSON object found; raw={text[:400]}"

    if not isinstance(d, dict):
        return None, f"Not a JSON object; raw={text[:400]}"
    if d.get("label") not in LABELS:
        return None, f"label not in allowed set: {d.get('label')!r}; raw={text[:400]}"
    return d, text


# ─── per-cell driver ───
def sweep_dir(results_dir: Path, agent_model_key: str, variant: str, clarify: bool,
              agent_prefix: str = "opencode") -> Path:
    suffix = "_clarify" if clarify else ""
    return results_dir / f"{agent_prefix}_{agent_model_key}_{variant}{suffix}"


def judge_one_cell(*, client: OpenAI, judge_model: str, results_dir: Path,
                   agent_model_key: str, variant: str, clarify: bool, slug: str,
                   manifest_entry: dict, max_traj_chars: int, overwrite: bool,
                   agent_prefix: str = "opencode", n_judges: int = 1) -> dict:
    sd = sweep_dir(results_dir, agent_model_key, variant, clarify, agent_prefix)
    cell = sd / slug
    out_path = cell / f"_audit.{judge_model}.json"
    if out_path.exists() and not overwrite:
        return {"status": "skipped", "path": str(out_path)}

    if not cell.is_dir():
        return {"status": "no_cell", "path": str(cell)}
    traj_p = cell / "_traj.json"
    sub_p = cell / "_submission.csv"
    if not traj_p.exists() and not sub_p.exists():
        return {"status": "no_traj_or_sub", "path": str(cell)}

    traj_text = pack_trajectory(traj_p, max_chars=max_traj_chars)
    sub_head = pack_submission_head(sub_p)
    clarify_d = None
    cl_path = cell / "_clarify.json"
    if clarify and cl_path.exists():
        try:
            clarify_d = json.loads(cl_path.read_text())
        except Exception:
            clarify_d = {"asked": None, "question": "<unreadable>", "answer": ""}

    sys_msg, user_msg = build_prompt(
        slug=slug, manifest_entry=manifest_entry,
        traj_text=traj_text, submission_head=sub_head, clarify=clarify_d,
    )
    prompt_sha = hashlib.sha256(
        (sys_msg + "\n---\n" + user_msg).encode()
    ).hexdigest()[:16]

    votes: list[dict] = []
    for _ in range(max(1, n_judges)):
        raw_i, usage_i = call_judge(client, judge_model, sys_msg, user_msg)
        parsed_i, _ = parse_judge_output(raw_i)
        votes.append({"raw": raw_i, "parsed": parsed_i, "usage": usage_i})

    if n_judges <= 1:
        raw = votes[0]["raw"]
        usage = votes[0]["usage"]
        parsed = votes[0]["parsed"]
    else:
        # Majority vote on label across parseable votes; tie-break by mean confidence.
        valid = [v for v in votes if v["parsed"] and v["parsed"].get("label") in LABELS]
        if not valid:
            raw = votes[0]["raw"]
            usage = votes[0]["usage"]
            parsed = None
        else:
            counts = Counter(v["parsed"]["label"] for v in valid)
            top = counts.most_common()
            top_n = top[0][1]
            tied = [lbl for lbl, c in top if c == top_n]
            if len(tied) == 1:
                winner = tied[0]
            else:
                def _mean_conf(lbl):
                    confs = [float(v["parsed"].get("confidence") or 0.0)
                             for v in valid if v["parsed"]["label"] == lbl]
                    return sum(confs) / max(1, len(confs))
                winner = max(tied, key=_mean_conf)
            winning_votes = [v for v in valid if v["parsed"]["label"] == winner]
            mean_conf = sum(float(v["parsed"].get("confidence") or 0.0)
                            for v in winning_votes) / len(winning_votes)
            evidence: list[str] = []
            for v in winning_votes:
                evidence.extend(v["parsed"].get("evidence_quotes") or [])
            parsed = {
                "label": winner,
                "confidence": round(mean_conf, 4),
                "evidence_quotes": evidence[:8],
                "rationale": winning_votes[0]["parsed"].get("rationale", ""),
                "vote_counts": dict(counts),
                "n_votes": len(valid),
            }
            raw = winning_votes[0]["raw"]
            # Aggregate token usage across all calls.
            keys = ("prompt_tokens", "completion_tokens", "total_tokens")
            usage = {k: sum(int(v["usage"].get(k) or 0) for v in votes) for k in keys}

    record = {
        "judge_model": judge_model,
        "agent_model_key": agent_model_key,
        "variant": variant,
        "clarify": clarify,
        "slug": slug,
        "prompt_sha256_16": prompt_sha,
        "manifest_metric_name": manifest_entry.get("metric_name"),
        "is_lower_better": manifest_entry.get("is_lower_better"),
        "n_judges": n_judges,
        "judge_votes": [
            {"raw": v["raw"], "parsed": v["parsed"], "usage": v["usage"]}
            for v in votes
        ] if n_judges > 1 else None,
        "judge_raw": raw,
        "judge_parsed": parsed,
        "usage": usage,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path.write_text(json.dumps(record, indent=2))
    return {"status": "ok" if parsed else "parse_error", "path": str(out_path),
            "label": (parsed or {}).get("label"), "conf": (parsed or {}).get("confidence")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True,
                    help="Benchmark directory (with results/)")
    ap.add_argument("--judge-model", required=True, help="Judge LLM")
    ap.add_argument("--agent-models", required=True,
                    help="Comma-separated list of agent_model keys "
                         "(the dir suffix in results/<agent>_<KEY>_*).")
    ap.add_argument("--agent-prefix", default="opencode",
                    help="Run-dir prefix (e.g. 'opencode' or 'claw'). "
                         "Defaults to 'opencode'.")
    ap.add_argument("--conditions", default="ambig_metric",
                    help="Comma-separated. Each item is one of: full, ambig_metric, "
                         "ambig_metric+clarify, full+clarify.")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--only-tasks", default=None,
                    help="Comma-separated slugs to restrict to.")
    ap.add_argument("--max-traj-chars", type=int, default=30_000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-judges", type=int, default=1,
                    help="Number of independent judge calls per cell. "
                         "If >1, the final label is the majority vote across calls "
                         "(ties broken by mean confidence).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--base-url",
                    default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = ap.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    results_dir = benchmark_dir / "results"
    if not results_dir.exists():
        sys.exit(f"results/ not found in {benchmark_dir}. Run step_2_run_agent.py first.")

    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    manifest = load_manifest(benchmark_dir)
    all_tasks = load_tasks(benchmark_dir)
    if args.only_tasks:
        wanted = set(t.strip() for t in args.only_tasks.split(",") if t.strip())
        tasks = [t for t in all_tasks if t in wanted]
    else:
        tasks = all_tasks
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    agent_models = [m.strip() for m in args.agent_models.split(",") if m.strip()]
    cond_specs = []
    for c in args.conditions.split(","):
        c = c.strip()
        if c.endswith("+clarify"):
            cond_specs.append((c[:-len("+clarify")], True))
        else:
            cond_specs.append((c, False))

    print(f"judge-model      : {args.judge_model}")
    print(f"agent-models     : {agent_models}")
    print(f"conditions       : {cond_specs}")
    print(f"tasks            : {len(tasks)}")
    print(f"concurrency      : {args.concurrency}")
    print(f"n-judges         : {args.n_judges}")
    print(f"overwrite        : {args.overwrite}")
    print()

    jobs = []
    for am in agent_models:
        for variant, clarify in cond_specs:
            for slug in tasks:
                if slug not in manifest:
                    print(f"  SKIP no-manifest: {slug}")
                    continue
                jobs.append((am, variant, clarify, slug))

    print(f"total cells: {len(jobs)}")
    n_ok = n_skip = n_err = n_no = 0
    t0 = time.time()

    def _run(job):
        am, variant, clarify, slug = job
        return job, judge_one_cell(
            client=client, judge_model=args.judge_model, results_dir=results_dir,
            agent_model_key=am, variant=variant, clarify=clarify, slug=slug,
            manifest_entry=manifest[slug],
            max_traj_chars=args.max_traj_chars, overwrite=args.overwrite,
            agent_prefix=args.agent_prefix, n_judges=args.n_judges,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_run, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                (am, variant, clarify, slug), res = fut.result()
            except Exception as e:
                print(f"  EX:                  {e}")
                n_err += 1
                continue
            tag = f"{am}/{variant}{'+clarify' if clarify else ''}/{slug}"
            st = res["status"]
            if st == "ok":
                n_ok += 1
                print(f"  ok    {tag}  label={res['label']}  conf={res['conf']}")
            elif st == "skipped":
                n_skip += 1
                print(f"  skip  {tag}")
            elif st in ("no_cell", "no_traj_or_sub"):
                n_no += 1
                print(f"  ---   {tag}  ({st})")
            else:
                n_err += 1
                print(f"  ERR   {tag}  ({st})")

    dt = time.time() - t0
    print()
    print(f"done. ok={n_ok} skip={n_skip} no_cell={n_no} err={n_err}  in {dt:.1f}s")


if __name__ == "__main__":
    main()
