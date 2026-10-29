#!/usr/bin/env python3
"""
BurnettBench — Factprobe on Full Ground Truth
Runs the factprobe pipeline against the expanded ground truth files
(*_groundtruth_full.jsonl) which contain both original + web-scraped facts.

Each fact carries an `in_v2` flag:
  in_v2=True  — fact is covered by the V2 document (recall study)
  in_v2=False — fact is NOT in V2 (out-of-scope accuracy study)

Outputs:
  burnettbench_{iid}_factprobes_full.jsonl
  burnettbench_{iid}_factprobe_results_full.jsonl
  burnettbench_{iid}_factprobe_verdicts_full.jsonl

Usage: python burnettbench_eval_factprobe_full.py [institution_id]
       (omit to run all institutions)
"""
import sys, json, time, re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

sys.path.insert(0, str(Path(__file__).parent))
from burnettbench_institutions import INSTITUTIONS

OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
DESK   = Path("/Users/eudaimonia/Desktop")

MODELS = {
    "qwen3-14b":       "qwen/qwen3-14b",
    "gpt-4o-mini":     "openai/gpt-4o-mini",
    "llama-3.1-70b":   "meta-llama/llama-3.1-70b-instruct",
    "claude-haiku-4-5":"anthropic/claude-haiku-4-5",
    "gemini-flash":    "google/gemini-3.8-flash",
}

PROBE_GEN_PROMPT = """\
Convert the following fact about {institution} into a single natural-language question \
that would elicit this fact as an answer. The question should be clear and direct. \
Return ONLY the question, nothing else.

Fact: {fact}"""

ACCURACY_JUDGE_PROMPT = """\
The following is a verified fact about {institution}:
FACT: "{fact}"

A user asked: "{question}"
An AI assistant responded: "{response}"

Did the AI correctly state this fact?
(A) CORRECT — response accurately and specifically states this fact
(B) INCORRECT — response states something wrong or contradictory about this fact
(C) NOT_ADDRESSED — response does not address this fact at all
Reply with exactly one letter: A, B, or C"""


def api_call(model_id, system, user_msg, temperature=0.7, max_tokens=400):
    for attempt in range(3):
        try:
            r = requests.post(OR_URL,
                headers={"Authorization": f"Bearer {OR_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model_id,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user",   "content": user_msg}],
                      "temperature": temperature, "max_tokens": max_tokens},
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


def run_institution(iid):
    inst    = INSTITUTIONS[iid]
    name    = inst["name"]
    v2_txt  = open(inst["llmstxt_v2"]).read()
    gt_file = DESK / f"burnettbench_{iid}_groundtruth_full.jsonl"

    if not gt_file.exists():
        print(f"  SKIP {iid}: no full groundtruth file — run expand+merge first")
        return

    facts = [json.loads(l) for l in open(gt_file) if l.strip()]

    probes_file   = DESK / f"burnettbench_{iid}_factprobes_full.jsonl"
    results_file  = DESK / f"burnettbench_{iid}_factprobe_results_full.jsonl"
    verdicts_file = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"

    in_v2_ids = {f["fact_id"] for f in facts if f.get("in_v2", True)}

    print(f"\n{'─'*65}")
    print(f"{name}: {len(facts)} facts ({len(in_v2_ids)} IN_V2, "
          f"{len(facts)-len(in_v2_ids)} OUT_V2)")

    # ── Step 1: Generate probe questions ──────────────────────────────────────
    existing_probes = {}
    if probes_file.exists():
        for line in open(probes_file):
            if line.strip():
                p = json.loads(line)
                existing_probes[p["fact_id"]] = p

    new_facts = [f for f in facts if f["fact_id"] not in existing_probes]
    print(f"  Probes: {len(existing_probes)} existing | {len(new_facts)} to generate")

    if new_facts:
        with open(probes_file, "a") as f:
            with ThreadPoolExecutor(max_workers=10) as ex:
                fmap = {
                    ex.submit(api_call,
                        "openai/gpt-4o-mini",
                        "You convert facts into questions. Return only the question.",
                        PROBE_GEN_PROMPT.format(institution=name, fact=fact["fact"]),
                        0, 100): fact
                    for fact in new_facts
                }
                for fut in as_completed(fmap):
                    fact = fmap[fut]
                    probe_q = fut.result() or ""
                    rec = {"fact_id":  fact["fact_id"],
                           "category": fact["category"],
                           "fact":     fact["fact"],
                           "probe":    probe_q,
                           "in_v2":    fact.get("in_v2", True)}
                    f.write(json.dumps(rec) + "\n")
                    existing_probes[fact["fact_id"]] = rec
        print(f"  Probes generated ✓")

    probes = list(existing_probes.values())

    # ── Step 2: Run models ─────────────────────────────────────────────────────
    done_run = load_done(results_file, ["model", "condition", "fact_id"])
    system_v2 = (f"You are a helpful assistant. Use the following reference "
                 f"information if relevant.\n<reference>\n{v2_txt}\n</reference>")
    system_a  = "You are a helpful assistant."

    tasks = [
        (mname, mid, cond, p)
        for mname, mid in MODELS.items()
        for cond in ["A", "V2"]
        for p in probes
        if (mname, cond, p["fact_id"]) not in done_run
    ]
    total = len(MODELS) * 2 * len(probes)
    print(f"  Runs: {total} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(api_call, mid,
                          system_v2 if cond == "V2" else system_a,
                          p["probe"]): (mname, cond, p)
                for mname, mid, cond, p in tasks
            }
            for fut in as_completed(futures):
                mname, cond, p = futures[fut]
                response = fut.result() or ""
                rec = {"institution": iid, "fact_id": p["fact_id"],
                       "category": p["category"], "fact": p["fact"],
                       "probe": p["probe"], "model": mname,
                       "condition": cond, "response": response,
                       "in_v2": p.get("in_v2", True)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"    {completed}/{len(tasks)} responses...", flush=True)
    print(f"  Runner done ✓")

    # ── Step 3: Judge accuracy ─────────────────────────────────────────────────
    responses  = [json.loads(l) for l in open(results_file) if l.strip()
                  and not json.loads(l)["response"].startswith("ERROR")]
    done_judge = load_done(verdicts_file, ["model", "condition", "fact_id"])
    judge_tasks = [r for r in responses
                   if (r["model"], r["condition"], r["fact_id"]) not in done_judge]
    print(f"  Judge: {len(judge_tasks)} tasks | {len(done_judge)} done")

    completed = 0
    with open(verdicts_file, "a") as f:
        with ThreadPoolExecutor(max_workers=15) as ex:
            fmap = {
                ex.submit(judge_call,
                    ACCURACY_JUDGE_PROMPT.format(
                        institution=name,
                        fact=r["fact"],
                        question=r["probe"],
                        response=r["response"])): r
                for r in judge_tasks
            }
            for fut in as_completed(fmap):
                r = fmap[fut]
                verdict = (fut.result() or "")[:1].upper()
                rec = {"institution": iid, "fact_id": r["fact_id"],
                       "category": r["category"], "fact": r["fact"],
                       "model": r["model"], "condition": r["condition"],
                       "verdict": verdict, "probe": r["probe"],
                       "in_v2": r.get("in_v2", True)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
    print(f"  Judge done ✓")

    # ── Summary ────────────────────────────────────────────────────────────────
    verdicts = [json.loads(l) for l in open(verdicts_file) if l.strip()]

    for label, subset in [("ALL FACTS", verdicts),
                           ("IN_V2 only", [v for v in verdicts if v.get("in_v2")]),
                           ("OUT_V2 only", [v for v in verdicts if not v.get("in_v2")])]:
        if not subset: continue
        print(f"\n  {'─'*55}")
        print(f"  {label}  (n={len(set(v['fact_id'] for v in subset))} facts)")
        print(f"  {'Model':<22}{'A':>9}{'V2':>9}{'LIFT':>8}")
        for m in MODELS:
            sc = {}
            for c in ["A", "V2"]:
                recs = [v for v in subset if v["model"]==m and v["condition"]==c]
                sc[c] = sum(1 for v in recs if v["verdict"]=="A") / len(recs) * 100 if recs else 0.0
            print(f"  {m:<22}{sc['A']:>8.1f}%{sc['V2']:>8.1f}%{sc['V2']-sc['A']:>+7.1f}%")
        sc = {}
        for c in ["A", "V2"]:
            recs = [v for v in subset if v["condition"]==c]
            sc[c] = sum(1 for v in recs if v["verdict"]=="A") / len(recs) * 100 if recs else 0.0
        print(f"  {'MEAN':<22}{sc['A']:>8.1f}%{sc['V2']:>8.1f}%{sc['V2']-sc['A']:>+7.1f}%")


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    targets = [t for t in targets if t in INSTITUTIONS]

    print(f"Factprobe (Full GT) — {', '.join(targets)}")
    for iid in targets:
        run_institution(iid)

    # Cross-institution grand summary
    if len(targets) > 1:
        print(f"\n{'='*65}")
        print("GRAND SUMMARY — ALL INSTITUTIONS")
        for label, flag in [("IN_V2 (recall)", True), ("OUT_V2 (out-of-scope)", False)]:
            print(f"\n  {label}")
            print(f"  {'Institution':<22}{'A':>9}{'V2':>9}{'LIFT':>8}")
            for iid in targets:
                vfile = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
                if not vfile.exists(): continue
                verdicts = [json.loads(l) for l in open(vfile) if l.strip()]
                subset   = [v for v in verdicts if v.get("in_v2") == flag]
                if not subset: continue
                sc = {}
                for c in ["A", "V2"]:
                    recs = [v for v in subset if v["condition"]==c]
                    sc[c] = sum(1 for v in recs if v["verdict"]=="A") / len(recs) * 100 if recs else 0.0
                name = INSTITUTIONS[iid]["name"]
                print(f"  {name:<22}{sc['A']:>8.1f}%{sc['V2']:>8.1f}%{sc['V2']-sc['A']:>+7.1f}%")


if __name__ == "__main__":
    main()
