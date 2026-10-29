#!/usr/bin/env python3
"""
BurnettBench — Gating Tuning Experiment
Tests two dimensions:
  1. OPT2 threshold sweep: 0.20, 0.25, 0.30, 0.40, 0.45  (0.35 = OPT2 already done)
  2. OPT3 top-k sweep: k=3, k=4, k=5                     (k=2  = OPT3 already done)

Runs locally via OpenRouter. No DGX needed.

Usage: python burnettbench_eval_gating_tune.py [institution_id]
"""
import sys, json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DESK   = Path("/Users/eudaimonia/Desktop")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

INSTITUTIONS = {
    "bakame": "Bakame AI",
    "mayo":   "Mayo Clinic",
    "ve":     "Vinson & Elkins",
    "uscis":  "USCIS",
    "rice":   "Rice University",
}
MODELS = {
    "qwen3-14b":        "qwen/qwen3-14b",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
    "llama-3.1-70b":    "meta-llama/llama-3.1-70b-instruct",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "gemini-flash":     "google/gemini-3.8-flash",
}

THRESHOLDS = [0.20, 0.25, 0.30, 0.40, 0.45, 0.75]
TOP_KS     = [3, 4, 5]

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


def parse_sections(v2_text):
    sections = []
    parts = re.split(r'\n(##[^#].*)', v2_text)
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if chunk.startswith('##'):
            header  = chunk.lstrip('#').strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            ann_match = re.search(r'\(use for:\s*(.+?)\)', header, re.IGNORECASE)
            annotation = ann_match.group(1) if ann_match else header.split('(')[0].strip()
            sections.append({
                "header":     header.split('(')[0].strip(),
                "annotation": annotation,
                "content":    content,
                "full":       f"## {header}\n{content}"
            })
            i += 2
        else:
            i += 1
    return [s for s in sections if "do not" not in s["header"].lower()
            and "disclose" not in s["header"].lower()]


class GatingClassifier:
    def __init__(self, sections):
        self.sections   = sections
        self.st_model   = SentenceTransformer("all-MiniLM-L6-v2")
        annotations     = [s["annotation"] for s in sections]
        self.bm25       = BM25Okapi([a.lower().split() for a in annotations])
        self.ann_embeds = self.st_model.encode(annotations, normalize_embeddings=True)

    def score(self, query):
        q_tokens    = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        bm25_max    = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
        q_embed    = self.st_model.encode([query], normalize_embeddings=True)[0]
        cos_scores = self.ann_embeds @ q_embed
        hybrid     = 0.5 * bm25_scores + 0.5 * cos_scores
        return float(hybrid.max()), hybrid


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


def cond_t(thresh): return f"T{int(thresh*100):02d}"
def cond_k(k):      return f"K{k}"


def run_institution(iid):
    name    = INSTITUTIONS[iid]
    v2_text = open(DESK / f"burnettbench_{iid}_llmstxt_v2.txt").read()
    probes  = [json.loads(l) for l in
               open(DESK / f"burnettbench_{iid}_factprobes_full.jsonl") if l.strip()]
    sections = parse_sections(v2_text)
    gate     = GatingClassifier(sections)

    results_file  = DESK / f"burnettbench_{iid}_gating_tune_results.jsonl"
    verdicts_file = DESK / f"burnettbench_{iid}_gating_tune_verdicts.jsonl"

    system_a  = "You are a helpful assistant."
    system_v2 = (f"You are a helpful assistant. Use the following reference "
                 f"information if relevant.\n<reference>\n{v2_text}\n</reference>")

    print(f"\n{'─'*65}")
    print(f"{name}: {len(probes)} probes | {len(sections)} sections")

    # Pre-compute scores
    probe_scores = {}
    for p in probes:
        max_s, hybrid = gate.score(p["probe"])
        probe_scores[p["fact_id"]] = {"max": max_s, "hybrid": hybrid}

    done_run = load_done(results_file, ["model", "condition", "fact_id"])
    tasks = []

    for mname, mid in MODELS.items():
        for p in probes:
            sc = probe_scores[p["fact_id"]]
            # Threshold sweep
            for thresh in THRESHOLDS:
                cond = cond_t(thresh)
                if (mname, cond, p["fact_id"]) not in done_run:
                    inject = sc["max"] >= thresh
                    sys = system_v2 if inject else system_a
                    tasks.append((mname, mid, cond, p, sys, float(sc["max"]), bool(inject)))
            # Top-k sweep
            for k in TOP_KS:
                cond = cond_k(k)
                if (mname, cond, p["fact_id"]) not in done_run:
                    top_secs = "\n\n".join(
                        sections[i]["full"]
                        for i in np.argsort(sc["hybrid"])[::-1][:k])
                    if top_secs:
                        sys = (f"You are a helpful assistant. Use the following "
                               f"reference information if relevant.\n<reference>\n"
                               f"{top_secs}\n</reference>")
                    else:
                        sys = system_a
                    tasks.append((mname, mid, cond, p, sys, float(sc["max"]), True))

    total = len(MODELS) * len(probes) * (len(THRESHOLDS) + len(TOP_KS))
    print(f"  Runs: {total} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(api_call, mid, sys, p["probe"]): (mname, cond, p, gs, gi)
                for mname, mid, cond, p, sys, gs, gi in tasks
            }
            for fut in as_completed(futures):
                mname, cond, p, gs, gi = futures[fut]
                response = fut.result() or ""
                rec = {"institution": iid, "fact_id": p["fact_id"],
                       "category": p["category"], "fact": p["fact"],
                       "probe": p["probe"], "model": mname,
                       "condition": cond, "response": response,
                       "in_v2": p.get("in_v2", True),
                       "gate_score": gs, "gate_inject": gi}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"    {completed}/{len(tasks)} responses...", flush=True)
    print("  Runner done ✓")

    # Judge
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
                    JUDGE_PROMPT.format(institution=name, fact=r["fact"],
                                        question=r["probe"], response=r["response"])): r
                for r in judge_tasks
            }
            for fut in as_completed(fmap):
                r = fmap[fut]
                verdict = (fut.result() or "")[:1].upper()
                rec = {"institution": iid, "fact_id": r["fact_id"],
                       "category": r["category"], "model": r["model"],
                       "condition": r["condition"], "verdict": verdict,
                       "in_v2": r.get("in_v2", True),
                       "gate_score": r.get("gate_score", 0),
                       "gate_inject": r.get("gate_inject", True)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
    print("  Judge done ✓")

    # Summary table
    all_files = [
        DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl",
        DESK / f"burnettbench_{iid}_gating_verdicts.jsonl",
        verdicts_file,
    ]
    all_v = []
    for fp in all_files:
        if fp.exists():
            all_v += [json.loads(l) for l in open(fp) if l.strip()]

    all_conds = (["A","V2","OPT2","OPT3"] +
                 [cond_t(t) for t in THRESHOLDS] +
                 [cond_k(k) for k in TOP_KS])

    for flag, label in [(True,"IN_V2"), (False,"OUT_V2")]:
        subset = [v for v in all_v if v.get("in_v2") == flag]
        if not subset: continue
        a_recs  = [v for v in subset if v["condition"]=="A"]
        v2_recs = [v for v in subset if v["condition"]=="V2"]
        a_acc   = sum(1 for v in a_recs  if v["verdict"]=="A")/len(a_recs)*100  if a_recs  else 0
        v2_acc  = sum(1 for v in v2_recs if v["verdict"]=="A")/len(v2_recs)*100 if v2_recs else 0
        print(f"\n  {label} (n={len(set(v['fact_id'] for v in subset))} facts)")
        print(f"  {'Cond':<10}{'Acc':>8}  {'vs V2':>8}  {'vs A':>8}")
        for cond in all_conds:
            recs = [v for v in subset if v["condition"]==cond]
            if not recs: continue
            acc = sum(1 for v in recs if v["verdict"]=="A")/len(recs)*100
            print(f"  {cond:<10}{acc:>7.1f}%  {acc-v2_acc:>+7.1f}pp  {acc-a_acc:>+7.1f}pp")


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    targets = [t for t in targets if t in INSTITUTIONS]
    print(f"GATING TUNE — thresholds {[cond_t(t) for t in THRESHOLDS]} | "
          f"top-k {[cond_k(k) for k in TOP_KS]}")
    for iid in targets:
        run_institution(iid)

    if len(targets) > 1:
        all_conds = (["A","V2","OPT2","OPT3"] +
                     [cond_t(t) for t in THRESHOLDS] +
                     [cond_k(k) for k in TOP_KS])

        for flag, label in [(False,"OUT_V2 accuracy"), (True,"IN_V2 recall")]:
            print(f"\n{'='*72}")
            print(f"GRAND SUMMARY — {label}")
            print(f"{'Institution':<22}" + "".join(f"{c:>7}" for c in all_conds))
            for iid in targets:
                all_files = [
                    DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl",
                    DESK / f"burnettbench_{iid}_gating_verdicts.jsonl",
                    DESK / f"burnettbench_{iid}_gating_tune_verdicts.jsonl",
                ]
                all_v = []
                for fp in all_files:
                    if fp.exists():
                        all_v += [json.loads(l) for l in open(fp) if l.strip()]
                subset = [v for v in all_v if v.get("in_v2") == flag]
                row    = f"{INSTITUTIONS[iid]:<22}"
                for cond in all_conds:
                    recs = [v for v in subset if v["condition"]==cond]
                    acc  = sum(1 for v in recs if v["verdict"]=="A")/len(recs)*100 if recs else 0.0
                    row += f"{acc:>6.1f}%"
                print(row)


if __name__ == "__main__":
    main()
