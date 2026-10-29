#!/usr/bin/env python3
"""
BurnettBench — Fix Experiment: Eliminating Out-of-Scope Accuracy Harm

Tests 3 modified V2 conditions designed to eliminate accuracy degradation
on questions outside the V2 document's scope, without hurting recall of
covered facts.

Current V2 system prompt (baseline):
  "You are a helpful assistant. Use the following reference information
   if relevant."

Fix 1 — Explicit release (system prompt only, doc unchanged):
  "...For questions not addressed in this document, rely on your general
   knowledge without reference to this document."

Fix 2 — Scope header prepended to V2 doc (system prompt unchanged):
  Adds "## Scope of This Document: covers X, Y, Z. For other topics,
   use general knowledge." at the top of V2.

Fix 3 — Both Fix 1 + Fix 2 combined.

Measures on BOTH IN_V2 and OUT_V2 facts:
  - Fix should eliminate OUT_V2 harm (bring it to ~0pp or positive)
  - Fix must not degrade IN_V2 recall

Usage: python burnettbench_eval_fix.py [institution_id]
       (omit to run all)
"""
import sys, json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import math, requests

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

FIX_CONDITIONS = ["FIX1", "FIX2", "FIX3"]

JUDGE_PROMPT = """\
The following is a verified fact about {institution}:
FACT: "{fact}"

A user asked: "{question}"
An AI assistant responded: "{response}"

Did the AI correctly state this fact?
(A) CORRECT — response accurately and specifically states this fact
(B) INCORRECT — response states something wrong or contradictory about this fact
(C) NOT_ADDRESSED — response does not address this fact at all
Reply with exactly one letter: A, B, or C"""

# ── Prompt builders ────────────────────────────────────────────────────────────

SYSTEM_BASE = (
    "You are a helpful assistant. Use the following reference information "
    "if relevant.\n<reference>\n{v2}\n</reference>"
)

SYSTEM_FIX1 = (
    "You are a helpful assistant. Use the following reference information "
    "for questions it directly addresses. "
    "For questions about topics not covered in this document, rely entirely "
    "on your general knowledge — do not force this document to answer "
    "questions it does not address.\n<reference>\n{v2}\n</reference>"
)

SCOPE_HEADER_TEMPLATE = """\
## Scope of This Document
This document covers the following topics only: {topics}.
For any question about topics not listed above, this document is not \
authoritative — answer from your general knowledge instead.

---
"""


def build_scope_header(v2_text):
    """Extract ## section titles (minus 'What We Do Not') for scope list."""
    headers = re.findall(r'^##\s+(.+)$', v2_text, re.MULTILINE)
    topics  = [h.split("(use for:")[0].strip()
               for h in headers
               if "do not" not in h.lower() and "disclose" not in h.lower()]
    return SCOPE_HEADER_TEMPLATE.format(topics=", ".join(topics))


def build_systems(v2_text):
    scope_header = build_scope_header(v2_text)
    v2_with_scope = scope_header + v2_text
    return {
        "FIX1": SYSTEM_FIX1.format(v2=v2_text),
        "FIX2": SYSTEM_BASE.format(v2=v2_with_scope),
        "FIX3": SYSTEM_FIX1.format(v2=v2_with_scope),
    }


# ── API helpers ────────────────────────────────────────────────────────────────

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


# ── Stats helpers ──────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (max(0, center-margin), min(1, center+margin), center)


def mcnemar_p(pairs):
    b = sum(1 for a, v in pairs if not a and v)
    c = sum(1 for a, v in pairs if a and not v)
    n = b + c
    if n == 0: return 1.0
    if n < 25:
        return 2 * min(stats.binom.cdf(min(b,c), n, 0.5),
                       1 - stats.binom.cdf(min(b,c)-1, n, 0.5))
    return stats.chi2.sf((abs(b-c)-1)**2/n, df=1)


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Main ───────────────────────────────────────────────────────────────────────

def run_institution(iid):
    inst      = INSTITUTIONS[iid]
    name      = inst["name"]
    v2_text   = open(inst["llmstxt_v2"]).read()
    systems   = build_systems(v2_text)

    probes_file   = DESK / f"burnettbench_{iid}_factprobes_full.jsonl"
    results_file  = DESK / f"burnettbench_{iid}_fix_results.jsonl"
    verdicts_file = DESK / f"burnettbench_{iid}_fix_verdicts.jsonl"

    if not probes_file.exists():
        print(f"  SKIP {iid}: run factprobe_full first")
        return

    probes = [json.loads(l) for l in open(probes_file) if l.strip()]
    print(f"\n{'─'*65}")
    print(f"{name}: {len(probes)} probes "
          f"({sum(1 for p in probes if p.get('in_v2'))} IN_V2, "
          f"{sum(1 for p in probes if not p.get('in_v2'))} OUT_V2)")

    # Print scope header for transparency
    scope = build_scope_header(v2_text)
    print(f"\n  Scope header generated:")
    for line in scope.split("\n")[:4]:
        print(f"    {line}")

    # ── Step 1: Run models under 3 fix conditions ─────────────────────────────
    done_run = load_done(results_file, ["model", "condition", "fact_id"])
    tasks = [
        (mname, mid, cond, p)
        for mname, mid in MODELS.items()
        for cond in FIX_CONDITIONS
        for p in probes
        if (mname, cond, p["fact_id"]) not in done_run
    ]
    total = len(MODELS) * len(FIX_CONDITIONS) * len(probes)
    print(f"\n  Runs: {total} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(api_call, mid, systems[cond], p["probe"]): (mname, cond, p)
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

    # ── Step 2: Judge ─────────────────────────────────────────────────────────
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
                    JUDGE_PROMPT.format(
                        institution=name, fact=r["fact"],
                        question=r["probe"], response=r["response"])): r
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

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    # Load original A and V2 verdicts for comparison
    orig_file = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
    orig = [json.loads(l) for l in open(orig_file) if l.strip()] if orig_file.exists() else []
    fix_verdicts = [json.loads(l) for l in open(verdicts_file) if l.strip()]
    all_verdicts = orig + fix_verdicts

    conds_ordered = ["A", "V2", "FIX1", "FIX2", "FIX3"]

    for label, flag in [("IN_V2 (recall — must stay high)", True),
                        ("OUT_V2 (scope harm — must recover)", False)]:
        subset = [v for v in all_verdicts if v.get("in_v2") == flag]
        if not subset: continue
        n_facts = len(set(v["fact_id"] for v in subset))
        print(f"\n  {label}  (n={n_facts} facts)")
        print(f"  {'Condition':<10}", end="")
        for m in list(MODELS.keys()) + ["MEAN"]:
            print(f"{m:>16}", end="")
        print()

        cond_means = {}
        for cond in conds_ordered:
            c_sub = [v for v in subset if v["condition"] == cond]
            if not c_sub: continue
            print(f"  {cond:<10}", end="")
            cond_accs = []
            for m in MODELS:
                recs = [v for v in c_sub if v["model"] == m]
                acc  = sum(1 for v in recs if v["verdict"]=="A") / len(recs) * 100 if recs else 0.0
                print(f"{acc:>15.1f}%", end="")
                cond_accs.append(acc)
            mean = sum(cond_accs) / len(cond_accs) if cond_accs else 0.0
            cond_means[cond] = mean
            print(f"{mean:>15.1f}%")

        # Delta vs V2 baseline
        if "V2" in cond_means and "A" in cond_means:
            print(f"\n  Δ vs V2 (positive = recovery):")
            for cond in ["A"] + FIX_CONDITIONS:
                if cond not in cond_means: continue
                delta = cond_means[cond] - cond_means["V2"]
                print(f"    {cond}: {delta:+.1f}pp", end="")
                # Significance vs V2
                v2_sub  = [v for v in subset if v["condition"]=="V2"]
                cnd_sub = [v for v in subset if v["condition"]==cond]
                v2_map  = {(v["model"], v["fact_id"]): v["verdict"]=="A" for v in v2_sub}
                cn_map  = {(v["model"], v["fact_id"]): v["verdict"]=="A" for v in cnd_sub}
                shared  = set(v2_map) & set(cn_map)
                pairs   = [(v2_map[k], cn_map[k]) for k in shared]
                p_val   = mcnemar_p(pairs) if pairs else 1.0
                print(f"  {sig(p_val)}")


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    targets = [t for t in targets if t in INSTITUTIONS]

    print("FIX EXPERIMENT — Out-of-Scope Accuracy Recovery")
    print("=" * 65)
    print("FIX1: modified system prompt (explicit release)")
    print("FIX2: scope header prepended to V2 document")
    print("FIX3: FIX1 + FIX2 combined\n")

    for iid in targets:
        run_institution(iid)

    # Grand summary across all institutions
    if len(targets) > 1:
        print(f"\n{'='*65}")
        print("GRAND SUMMARY — OUT_V2 recovery across all institutions")
        print(f"{'Institution':<22}{'A':>8}{'V2':>8}{'FIX1':>8}{'FIX2':>8}{'FIX3':>8}")
        for iid in targets:
            vf_orig = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
            vf_fix  = DESK / f"burnettbench_{iid}_fix_verdicts.jsonl"
            if not vf_orig.exists() or not vf_fix.exists(): continue
            orig = [json.loads(l) for l in open(vf_orig) if l.strip()]
            fix  = [json.loads(l) for l in open(vf_fix)  if l.strip()]
            all_v = orig + fix
            out   = [v for v in all_v if not v.get("in_v2")]
            row   = f"{INSTITUTIONS[iid]['name']:<22}"
            for cond in ["A", "V2", "FIX1", "FIX2", "FIX3"]:
                recs = [v for v in out if v["condition"]==cond]
                acc  = sum(1 for v in recs if v["verdict"]=="A") / len(recs) * 100 if recs else 0.0
                row += f"{acc:>7.1f}%"
            print(row)

        print(f"\nIN_V2 recall — must not drop:")
        print(f"{'Institution':<22}{'A':>8}{'V2':>8}{'FIX1':>8}{'FIX2':>8}{'FIX3':>8}")
        for iid in targets:
            vf_orig = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
            vf_fix  = DESK / f"burnettbench_{iid}_fix_verdicts.jsonl"
            if not vf_orig.exists() or not vf_fix.exists(): continue
            orig = [json.loads(l) for l in open(vf_orig) if l.strip()]
            fix  = [json.loads(l) for l in open(vf_fix)  if l.strip()]
            all_v = orig + fix
            inv   = [v for v in all_v if v.get("in_v2")]
            row   = f"{INSTITUTIONS[iid]['name']:<22}"
            for cond in ["A", "V2", "FIX1", "FIX2", "FIX3"]:
                recs = [v for v in inv if v["condition"]==cond]
                acc  = sum(1 for v in recs if v["verdict"]=="A") / len(recs) * 100 if recs else 0.0
                row += f"{acc:>7.1f}%"
            print(row)


if __name__ == "__main__":
    main()
