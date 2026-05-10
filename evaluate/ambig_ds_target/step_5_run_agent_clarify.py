#!/usr/bin/env python3
"""Step 5 — run the agent on Ambig-DS-T WITH one-turn target clarification.

Mirrors evaluate/ambig_ds_metric/step_3_run_agent_clarify.py but for the
target-ambiguity benchmark. Per (slug, variant) we run the agent in three
phases:

  Phase A (ASK):    agent gets the prompt + a clarify-protocol footer
                    instructing it to write ONE clarifying question
                    (or `NONE`) to ./_question.txt and stop. No submission
                    instructions are given in this phase.
  Phase B (ANSWER): if the agent asked, an answerer LLM is given the per-task
                    decoy manifest (which of val_1/val_2 is real) plus the
                    question, and produces a 1-2 sentence answer.
  Phase C (SOLVE):  a fresh workspace is built; the same prompt + the [Q,A]
                    transcript are passed; runs to completion exactly like
                    step_4_run_agent.py and is graded by the per-task eval.py.

Only `--variant ambig_target` is meaningful here (in the `full` variant the
agent already knows the real target column; we keep it as an option for
ablation symmetry with ambig_ds_metric).

Prerequisites:
  - step_1_setup_benchmark.py has been run
  - OPENAI_API_KEY (or pass --api-key)
  - opencode on PATH (or pass --agent-bin)

Usage:
    python step_5_run_agent_clarify.py --benchmark-dir ./benchmark \\
        --variant ambig_target --model anthropic_claude_haiku_4_5_v1_0 \\
        --answerer-model anthropic_claude_haiku_4_5_v1_0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from agents import default_bin
from clarify_answerer import answer_clarify_target
from step_4_run_agent import (
    SUBMISSION_NAME,
    VARIANTS,
    build_workspace,
    find_submission,
    grade,
    load_tasks,
    run_agent,
    submission_shape,
)


# ---------- prompts ----------
CLARIFY_PROTOCOL = """

[CLARIFY PROTOCOL — read carefully]
Before doing any modeling, you may ask exactly ONE clarifying question to
the user.

  - You MAY first do a quick inspection of the data to inform your question
    — e.g. read column headers, peek at a few rows, check basic shapes /
    dtypes, look at the candidate target columns. Keep this brief.
  - Then write your question (or the literal word `NONE`) to:
        {qfile}
  - Keep the question under 50 words and self-contained (the user only sees
    that file, not your tool calls or notes).
  - After writing the file, STOP. Do NOT train a model or write a submission
    in this turn. The user will reply in the next turn with the answer
    (or with `NONE`).
"""

# Stricter ("conservative") version: discourages unnecessary clarification on
# already-specified tasks. Selected with --strict-protocol.
STRICT_CLARIFY_PROTOCOL = """

[CLARIFY PROTOCOL — read carefully]
Ask only if the task cannot be solved from the prompt and data. If the
task is fully specified or the answer can be inferred from local evidence,
write NONE. Unnecessary clarification is penalized.

  - You MAY first do a quick inspection of the data to inform your decision
    — e.g. read column headers, peek at a few rows, check basic shapes /
    dtypes, look at the candidate target columns. Keep this brief.
  - Then write your question (or the literal word `NONE`) to:
        {qfile}
  - Keep the question under 50 words and self-contained (the user only sees
    that file, not your tool calls or notes).
  - After writing the file, STOP. Do NOT train a model or write a submission
    in this turn. The user will reply in the next turn with the answer
    (or with `NONE`).
"""

CLARIFY_INJECTION = """

[CLARIFICATION FROM USER]
Earlier you asked:
> {question}

The user replied:
> {answer}

Now produce the submission as instructed above.
"""

NO_CLARIFICATION_NOTE = """

[NOTE]
You did not use your one clarifying question (or wrote NONE). Proceed
directly with the modeling task as instructed above.
"""


# --------------------------------------------------------------------------- #
def _load_task_manifest(benchmark_dir: Path, slug: str) -> dict | None:
    """Per-task decoy manifest (true_target_column, original_target_name, ...).

    Prefers the rebuilt ambig manifest (flat schema).  Falls back to the HF
    release manifest (restructured schema) and normalises the keys so that
    callers can always access ``manifest["true_target_column"]``.
    """
    for cand in [
        benchmark_dir / "data" / slug / "ambig" / "_manifest.json",
        benchmark_dir / "release" / "tasks" / slug / "_manifest.json",
    ]:
        if cand.exists():
            try:
                m = json.loads(cand.read_text())
            except Exception:
                continue
            # Normalise restructured HF release schema → flat keys.
            if "task" in m and isinstance(m["task"], dict):
                t = m["task"]
                m.setdefault("true_target_column",
                             t.get("true_target_column_in_ambig"))
                m.setdefault("decoy_column",
                             t.get("decoy_column_in_ambig"))
                m.setdefault("original_target_name",
                             t.get("original_target_name"))
                m.setdefault("target_type", t.get("task_type"))
            return m
    return None


def _build_ask_prompt(slug: str, variant: str, benchmark_dir: Path,
                      qfile: Path, strict_protocol: bool = False) -> str:
    """Phase A prompt: raw task body + ASK footer (NO submission footer).

    If ``strict_protocol`` is True, swap in ``STRICT_CLARIFY_PROTOCOL``
    ("conservative" ask policy from the paper). Default False preserves the
    permissive policy used by all previously completed runs.
    """
    if variant == "full":
        prompt_file = benchmark_dir / "release" / "tasks" / slug / "task.txt"
    else:
        prompt_file = benchmark_dir / "release" / "tasks" / slug / "task_ambig.txt"
    base = prompt_file.read_text()
    proto = STRICT_CLARIFY_PROTOCOL if strict_protocol else CLARIFY_PROTOCOL
    ask_footer = (
        f"\n\n---\n\n"
        f"## Task instructions\n\n"
        f"You are a data scientist. The dataset for this competition is in `./data/`.\n"
        f"Do NOT write a submission yet — see the clarify protocol below."
        + proto.format(qfile=str(qfile))
    )
    return base + ask_footer


# --------------------------------------------------------------------------- #
def run_one_clarify(slug: str, variant: str, model: str, args, run_dir: Path,
                    benchmark_dir: Path) -> dict:
    log: dict = {"slug": slug, "variant": variant, "model": model,
                 "started_at": datetime.now().isoformat()}
    out_task = run_dir / slug
    out_task.mkdir(parents=True, exist_ok=True)
    grade_file = out_task / "_grade.json"
    clarify_file = out_task / "_clarify.json"
    if args.skip_existing and grade_file.exists():
        return {**log, "status": "skipped_existing"}
    # In --clarify-only mode no grade is written; treat existing _clarify.json as done.
    if args.skip_existing and getattr(args, "clarify_only", False) and clarify_file.exists():
        return {**log, "status": "skipped_existing"}

    manifest = _load_task_manifest(benchmark_dir, slug)
    if manifest is None:
        return {**log, "status": "manifest_missing"}

    # ---------- Phase A: ASK ----------
    ask_run_name = f"{args.run_name}/_ask"
    try:
        ws_ask, _ = build_workspace(
            slug, variant, benchmark_dir,
            benchmark_dir / "workspaces", ask_run_name)
    except Exception as e:
        return {**log, "status": "build_failed_ask", "error": str(e)}

    qfile = ws_ask / "_question.txt"
    ask_prompt = _build_ask_prompt(
        slug, variant, benchmark_dir, qfile,
        strict_protocol=getattr(args, "strict_protocol", False),
    )
    (ws_ask / "task.md").write_text(ask_prompt)

    if args.dry_run:
        msgA, toolsA, itersA, costA = "DRY_RUN", [], 0, ""
        elapsedA = 0.0
        timed_out_A = False
        question = ""
        asked = False
    else:
        t0 = time.time()
        msgA, toolsA, itersA, costA = run_agent(
            args.agent_bin, model, ask_prompt, ws_ask,
            args.api_key, args.base_url, timeout=args.ask_timeout)
        elapsedA = time.time() - t0
        timed_out_A = isinstance(msgA, str) and msgA.startswith("ERROR: timeout")

        question = ""
        if qfile.exists():
            try:
                question = qfile.read_text().strip()
            except Exception:
                question = ""
        asked = bool(question) and question.strip().upper() != "NONE"

    # ---------- Phase B: ANSWER ----------
    answer_text = ""
    refused = False
    answerer_error = None
    if asked:
        try:
            ans = answer_clarify_target(
                question=question,
                task_name=slug,
                true_target_column=manifest["true_target_column"],
                original_target_name=manifest.get("original_target_name", ""),
                target_type=manifest.get("target_type", "unknown"),
                model=args.answerer_model,
                api_key=args.api_key,
                base_url=args.base_url,
                variant=variant,
            )
            answer_text = ans["answer"]
            refused = ans["refused"]
        except Exception as e:
            answerer_error = f"{type(e).__name__}: {e}"
            answer_text = f"(answerer error: {answerer_error})"

    (out_task / "_clarify.json").write_text(json.dumps({
        "slug": slug, "variant": variant, "model": model,
        "answerer_model": args.answerer_model,
        "asked": asked, "refused": refused,
        "answerer_error": answerer_error,
        "question": question, "answer": answer_text,
        "ask_iters": itersA, "ask_time_sec": round(elapsedA, 1),
        "ask_cost": costA, "ask_timed_out": timed_out_A,
        "strict_protocol": bool(getattr(args, "strict_protocol", False)),
        "clarify_only": bool(getattr(args, "clarify_only", False)),
    }, indent=2))

    if itersA > 0:
        shutil.rmtree(ws_ask, ignore_errors=True)

    # If --clarify-only, stop here. Skip Phase C (no solve, no submission, no grade).
    if getattr(args, "clarify_only", False):
        log.update({
            "status": "ask_only_done",
            "ask_time_sec": round(elapsedA, 1),
            "ask_iters": itersA,
            "ask_cost": costA,
            "asked": asked,
            "answerer_refused": refused,
            "strict_protocol": bool(getattr(args, "strict_protocol", False)),
        })
        return log

    # ---------- Phase C: SOLVE ----------
    try:
        ws, prompt_text = build_workspace(
            slug, variant, benchmark_dir,
            benchmark_dir / "workspaces", args.run_name)
    except Exception as e:
        return {**log, "status": "build_failed_solve", "error": str(e),
                "asked": asked, "ask_iters": itersA}

    injection = (CLARIFY_INJECTION.format(question=question, answer=answer_text)
                 if asked else NO_CLARIFICATION_NOTE)
    solve_prompt = prompt_text + injection
    (ws / "task.md").write_text(solve_prompt)

    if args.dry_run:
        print(f"\n--- DRY RUN: {slug} ({variant}) [+clarify] ---")
        print(f"workspace: {ws}")
        print(f"asked={asked}  question={question!r}  answer={answer_text!r}")
        print(f"solve prompt ({len(solve_prompt)} chars):")
        print(solve_prompt[:1500] + ("..." if len(solve_prompt) > 1500 else ""))
        return {**log, "status": "dry_run", "asked": asked, "ask_iters": itersA}

    t1 = time.time()
    msgC, toolsC, itersC, costC = run_agent(
        args.agent_bin, model, solve_prompt, ws,
        args.api_key, args.base_url, timeout=args.timeout)
    elapsedC = time.time() - t1

    timed_out_C = isinstance(msgC, str) and msgC.startswith("ERROR: timeout")

    traj = {
        "slug": slug, "variant": variant, "model": model,
        "elapsed_sec": elapsedA + elapsedC,
        "iterations": itersC, "cost": costC,
        "summary": msgC, "tool_uses": toolsC,
        "timed_out": timed_out_C,
        "ask_phase": {
            "summary": msgA, "iterations": itersA, "cost": costA,
            "time_sec": round(elapsedA, 1), "tool_uses": toolsA,
            "timed_out": timed_out_A, "asked": asked,
        },
    }
    (out_task / "_traj.json").write_text(json.dumps(traj, indent=2))

    # Dump tool-call inputs into the workspace so that step_7_target_audit
    # can classify which target the agent used even when the agent ran code
    # inline (via tool calls) without writing .py files to the workspace.
    tool_code = "\n".join(
        tu.get("input", "") for tu in toolsC if tu.get("input"))
    if tool_code.strip():
        (ws / "_tool_calls.txt").write_text(tool_code)

    sub_in_ws = find_submission(ws)
    if sub_in_ws is not None:
        sub_dest = out_task / SUBMISSION_NAME
        shutil.copy2(sub_in_ws, sub_dest)
        shape = submission_shape(sub_dest, slug, benchmark_dir)
        (out_task / "_shape.json").write_text(json.dumps(shape, indent=2, default=str))
        report = grade(sub_dest, slug, benchmark_dir, out_task)
    else:
        (out_task / "_shape.json").write_text(json.dumps(
            {"error": "no submission found"}, indent=2))
        report = {"error": "no submission found", "submission_exists": False}

    # Ensure invalid / errored runs score 0 and remain in the denominator.
    report.setdefault("score", 0)
    report.setdefault("score_rpg", 0.0)

    grade_file.write_text(json.dumps(report, indent=2, default=str))

    log.update({
        "status": "ok" if "error" not in report else "graded_with_error",
        "elapsed_sec": round(elapsedA + elapsedC, 1),
        "ask_time_sec": round(elapsedA, 1),
        "solve_time_sec": round(elapsedC, 1),
        "iterations": itersC, "ask_iters": itersA,
        "cost": costC,
        "score": report.get("score"),
        "asked": asked, "answerer_refused": refused,
    })
    return log


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="ambig_target")
    p.add_argument("--model", required=True)
    p.add_argument("--answerer-model", default=None,
                   help="LLM that answers the agent's clarifying question "
                        "(default: same as --model)")
    p.add_argument("--tasks", default="all")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Phase C (solve) timeout per task, seconds")
    p.add_argument("--ask-timeout", type=int, default=120,
                   help="Phase A (ask) timeout per task, seconds")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--agent-bin", default=None,
                   help="Path to opencode binary (default: auto-detect)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url",
                   default=os.environ.get("OPENAI_BASE_URL",
                                          "https://api.openai.com/v1"))
    p.add_argument("--run-name", default=None,
                   help="Run dir name (default: <agent>_<model>_<variant>_clarify"
                        " [+'_strict' if --strict-protocol] [+'_ask_only' if --clarify-only])")
    p.add_argument("--strict-protocol", dest="strict_protocol", action="store_true",
                   help="Use the conservative ('strict') ask protocol from the paper "
                        "(STRICT_CLARIFY_PROTOCOL). Default: permissive.")
    p.add_argument("--clarify-only", dest="clarify_only", action="store_true",
                   help="Run only Phase A (ask) + Phase B (answer). Skip Phase C (solve), "
                        "do not produce a submission or grade.")
    args = p.parse_args()

    if args.agent_bin is None:
        args.agent_bin = default_bin()
    if args.answerer_model is None:
        args.answerer_model = args.model

    benchmark_dir = args.benchmark_dir.resolve()
    if not (benchmark_dir / "task_list.txt").exists():
        sys.exit(f"task_list.txt not found in {benchmark_dir}. "
                 "Run step_1_setup_benchmark.py first.")

    args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.api_key and not args.dry_run:
        sys.exit("No API key. Set OPENAI_API_KEY or pass --api-key.")

    args.run_name = (args.run_name
                     or (f"opencode_{args.model}_{args.variant}"
                         + ("_ask_only" if args.clarify_only else "_clarify")
                         + ("_strict" if args.strict_protocol else "")))
    run_dir = benchmark_dir / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = load_tasks(benchmark_dir)
    if args.tasks == "all":
        tasks = all_tasks
    else:
        wanted = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in all_tasks if t in wanted]
        missing = wanted - set(tasks)
        if missing:
            print(f"WARNING: unknown tasks: {missing}")

    print(f"=== {args.run_name} === {len(tasks)} tasks")
    print(f"  ask_timeout={args.ask_timeout}s  solve_timeout={args.timeout}s")
    print(f"  answerer={args.answerer_model}")
    print(f"  results -> {run_dir}")
    runlog = run_dir / "_runlog.jsonl"

    with runlog.open("a") as fh:
        for i, slug in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {slug}")
            log = run_one_clarify(slug, args.variant, args.model, args,
                                  run_dir, benchmark_dir)
            fh.write(json.dumps(log, default=str) + "\n")
            fh.flush()
            status = log.get("status", "?")
            score = log.get("score")
            score_s = f"{score:.4g}" if isinstance(score, (int, float)) else "—"
            print(
                f"  status={status}  score={score_s}  asked={log.get('asked','—')}  "
                f"ask={log.get('ask_time_sec','—')}s/{log.get('ask_iters','—')}it  "
                f"solve={log.get('solve_time_sec','—')}s/{log.get('iterations','—')}it"
            )

    print(f"\nDone. Runlog: {runlog}")


if __name__ == "__main__":
    main()
