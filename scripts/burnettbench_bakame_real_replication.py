#!/usr/bin/env python3
"""
Replication: Real Published Bakame llms.txt vs. Synthetic V2
Runs the same factprobe pipeline on Bakame's actual published llms.txt
(bakame.online/llms.txt) to test whether findings hold on a real-world file.

Conditions:
  A       — no context (same as original)
  REAL_V2 — real published llms.txt (new)

Compares results to original synthetic V2 experiment.
"""
import json, time, re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

DESK   = Path("/Users/eudaimonia/Desktop")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "qwen3-14b":        "qwen/qwen3-14b",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
    "llama-3.1-70b":    "meta-llama/llama-3.1-70b-instruct",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "gemini-flash":     "google/gemini-3.8-flash",
}

COVERAGE_PROMPT = """\
The following is a fact about Bakame AI:
FACT: "{fact}"

Does the following document explicitly cover this fact (i.e., does it contain \
information that would allow an AI to answer a question about this fact)?

DOCUMENT:
{doc}

Reply with exactly one word: YES or NO"""

JUDGE_PROMPT = """\
The following is a verified fact about Bakame AI:
FACT: "{fact}"

A user asked: "{question}"
An AI assistant responded: "{response}"

Did the AI correctly state this fact?
(A) CORRECT — response accurately and specifically states this fact
(B) INCORRECT — response states something wrong or contradictory about this fact
(C) NOT_ADDRESSED — response does not address this fact at all
Reply with exactly one letter: A, B, or C"""


def api_call(model_id, system, user_msg):
    for attempt in range(3):
        try:
            r = requests.post(OR_URL,
                headers={"Authorization": f"Bearer {OR_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model_id,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user",   "content": user_msg}],
                      "temperature": 0.7, "max_tokens": 400},
                timeout=60)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except Exception as e:
            if attempt == 2: return f"ERROR: {e}"
            time.sleep(2 ** attempt)


def judge_call(prompt_text):
    for attempt in range(3):
        try:
            r = requests.post(OR_URL,
                headers={"Authorization": f"Bearer {OR_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "openai/gpt-4o-mini",
                      "messages": [{"role": "user", "content": prompt_text}],
                      "temperature": 0, "max_tokens": 10},
                timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2: return "ERROR"
            time.sleep(2 ** attempt)


def load_done(path, keys):
    if not path.exists(): return set()
    done = set()
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            done.add(tuple(r[k] for k in keys))
    return done


def main():
    real_v2 = open(DESK / "burnettbench_bakame_llmstxt_real.txt").read()
    probes  = [json.loads(l) for l in
               open(DESK / "burnettbench_bakame_factprobes_full.jsonl") if l.strip()]

    print(f"Bakame Real llms.txt Replication")
    print(f"  {len(probes)} probe questions")
    print(f"  Real llms.txt: {len(real_v2)} chars")

    # ── Step 1: Classify facts against real llms.txt ──────────────────────────
    coverage_file = DESK / "burnettbench_bakame_real_coverage.jsonl"
    existing_cov  = {}
    if coverage_file.exists():
        for line in open(coverage_file):
            if line.strip():
                r = json.loads(line)
                existing_cov[r["fact_id"]] = r["in_real"]

    new_probes = [p for p in probes if p["fact_id"] not in existing_cov]
    print(f"\n  Coverage classification: {len(existing_cov)} done | {len(new_probes)} remaining")

    if new_probes:
        with open(coverage_file, "a") as f:
            with ThreadPoolExecutor(max_workers=10) as ex:
                fmap = {
                    ex.submit(judge_call,
                        COVERAGE_PROMPT.format(fact=p["fact"], doc=real_v2[:8000])): p
                    for p in new_probes
                }
                for fut in as_completed(fmap):
                    p = fmap[fut]
                    answer = (fut.result() or "").strip().upper()
                    in_real = answer.startswith("Y")
                    existing_cov[p["fact_id"]] = in_real
                    f.write(json.dumps({"fact_id": p["fact_id"], "fact": p["fact"],
                                        "in_real": in_real}) + "\n")

    in_real_ids  = {fid for fid, v in existing_cov.items() if v}
    out_real_ids = {fid for fid, v in existing_cov.items() if not v}
    print(f"  IN_REAL: {len(in_real_ids)} facts | OUT_REAL: {len(out_real_ids)} facts")

    # Compare with synthetic V2 coverage
    in_synth  = {p["fact_id"] for p in probes if p.get("in_v2")}
    out_synth = {p["fact_id"] for p in probes if not p.get("in_v2")}
    newly_covered = in_real_ids - in_synth
    lost_coverage = in_synth - in_real_ids
    print(f"\n  vs. Synthetic V2:")
    print(f"    Newly covered by real llms.txt: {len(newly_covered)} facts")
    print(f"    In synthetic but not real:      {len(lost_coverage)} facts")
    if newly_covered:
        for fid in list(newly_covered)[:3]:
            p = next(x for x in probes if x["fact_id"]==fid)
            print(f"      + {p['fact'][:80]}")

    # ── Step 2: Run models ─────────────────────────────────────────────────────
    results_file  = DESK / "burnettbench_bakame_real_results.jsonl"
    verdicts_file = DESK / "burnettbench_bakame_real_verdicts.jsonl"

    system_real = (f"You are a helpful assistant. Use the following reference "
                   f"information if relevant.\n<reference>\n{real_v2}\n</reference>")
    system_a    = "You are a helpful assistant."

    done_run = load_done(results_file, ["model", "condition", "fact_id"])
    tasks = []
    for mname, mid in MODELS.items():
        for p in probes:
            for cond in ["A", "REAL_V2"]:
                if (mname, cond, p["fact_id"]) not in done_run:
                    sys = system_real if cond == "REAL_V2" else system_a
                    tasks.append((mname, mid, cond, p, sys))

    total = len(MODELS) * 2 * len(probes)
    print(f"\n  Runs: {total} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(api_call, mid, sys, p["probe"]): (mname, cond, p)
                for mname, mid, cond, p, sys in tasks
            }
            for fut in as_completed(futures):
                mname, cond, p = futures[fut]
                response = fut.result() or ""
                in_real  = existing_cov.get(p["fact_id"], False)
                rec = {"fact_id": p["fact_id"], "category": p["category"],
                       "fact": p["fact"], "probe": p["probe"],
                       "model": mname, "condition": cond, "response": response,
                       "in_v2": p.get("in_v2"), "in_real": in_real}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 50 == 0:
                    print(f"    {completed}/{len(tasks)} responses...", flush=True)
    print("  Runner done ✓")

    # ── Step 3: Judge ──────────────────────────────────────────────────────────
    responses   = [json.loads(l) for l in open(results_file) if l.strip()
                   and not json.loads(l)["response"].startswith("ERROR")]
    done_judge  = load_done(verdicts_file, ["model", "condition", "fact_id"])
    judge_tasks = [r for r in responses
                   if (r["model"], r["condition"], r["fact_id"]) not in done_judge]
    print(f"  Judge: {len(judge_tasks)} tasks | {len(done_judge)} done")

    completed = 0
    with open(verdicts_file, "a") as f:
        with ThreadPoolExecutor(max_workers=15) as ex:
            fmap = {
                ex.submit(judge_call,
                    JUDGE_PROMPT.format(fact=r["fact"], question=r["probe"],
                                        response=r["response"])): r
                for r in judge_tasks
            }
            for fut in as_completed(fmap):
                r = fmap[fut]
                verdict = (fut.result() or "")[:1].upper()
                rec = {"fact_id": r["fact_id"], "category": r["category"],
                       "fact": r["fact"], "model": r["model"],
                       "condition": r["condition"], "verdict": verdict,
                       "in_v2": r.get("in_v2"), "in_real": r.get("in_real")}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 50 == 0:
                    print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
    print("  Judge done ✓")

    # ── Summary ────────────────────────────────────────────────────────────────
    verdicts = [json.loads(l) for l in open(verdicts_file) if l.strip()]
    orig_v   = [json.loads(l) for l in
                open(DESK / "burnettbench_bakame_factprobe_verdicts_full.jsonl") if l.strip()]

    def acc(vs, cond, key=None, val=None):
        v = [x for x in vs if x["condition"]==cond]
        if key: v = [x for x in v if x.get(key)==val]
        return round(sum(1 for x in v if x["verdict"]=="A")/len(v)*100, 1) if v else 0.0

    print(f"\n{'─'*60}")
    print(f"RESULTS COMPARISON — Bakame AI")
    print(f"\n  {'Subset':<20}{'Cond':<12}{'Synthetic V2':<16}{'Real llms.txt'}")
    print(f"  {'─'*60}")

    # IN_V2 vs IN_REAL (covered facts)
    a_orig   = acc(orig_v,  "A",  "in_v2", True)
    v2_orig  = acc(orig_v,  "V2", "in_v2", True)
    a_real   = acc(verdicts, "A",       "in_real", True)
    rv2_real = acc(verdicts, "REAL_V2", "in_real", True)
    print(f"  {'IN-scope facts':<20}{'A':<12}{a_orig:<16}{a_real}")
    print(f"  {'IN-scope facts':<20}{'V2/REAL_V2':<12}{v2_orig:<16}{rv2_real}")
    print(f"  {'IN-scope facts':<20}{'Lift':<12}{round(v2_orig-a_orig,1):<16}{round(rv2_real-a_real,1)}")

    print()
    # OUT_V2 vs OUT_REAL (uncovered facts)
    a_orig_o   = acc(orig_v,  "A",  "in_v2", False)
    v2_orig_o  = acc(orig_v,  "V2", "in_v2", False)
    a_real_o   = acc(verdicts, "A",       "in_real", False)
    rv2_real_o = acc(verdicts, "REAL_V2", "in_real", False)
    print(f"  {'OUT-scope facts':<20}{'A':<12}{a_orig_o:<16}{a_real_o}")
    print(f"  {'OUT-scope facts':<20}{'V2/REAL_V2':<12}{v2_orig_o:<16}{rv2_real_o}")
    print(f"  {'OUT-scope facts':<20}{'Harm':<12}{round(v2_orig_o-a_orig_o,1):<16}{round(rv2_real_o-a_real_o,1)}")

    print(f"\n  KEY: does real llms.txt show same pattern?")
    in_lift  = round(rv2_real - a_real, 1)
    out_harm = round(rv2_real_o - a_real_o, 1)
    print(f"    IN-scope lift:  {in_lift:+.1f}pp  ({'✓ replicates' if in_lift > 30 else '✗ does not replicate'})")
    print(f"    OUT-scope harm: {out_harm:+.1f}pp  ({'✓ replicates' if out_harm < -2 else '~ weak or no harm'})")


if __name__ == "__main__":
    main()
