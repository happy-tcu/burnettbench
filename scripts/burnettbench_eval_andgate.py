#!/usr/bin/env python3
"""
BurnettBench — AND Gate Experiment
Tests a stricter gating logic: inject only when BOTH BM25 AND cosine
independently exceed a threshold, rather than their average.

Conditions tested (all inject full V2 doc when gate passes):
  AND30 — BM25 >= 0.30 AND cosine >= 0.30
  AND35 — BM25 >= 0.35 AND cosine >= 0.35
  AND40 — BM25 >= 0.40 AND cosine >= 0.40
  AND45 — BM25 >= 0.45 AND cosine >= 0.45

Compare against existing A, V2, OPT2 (avg hybrid @ 0.35) baselines.

Usage: python burnettbench_eval_andgate.py [institution_id]
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
STUDY  = Path("/Users/eudaimonia/Desktop/burnettbench_study2")
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

AND_THRESHOLDS = [0.30, 0.35, 0.40, 0.45]

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
            header    = chunk.lstrip('#').strip()
            content   = parts[i+1].strip() if i+1 < len(parts) else ""
            ann_match = re.search(r'\(use for:\s*(.+?)\)', header, re.IGNORECASE)
            annotation = ann_match.group(1) if ann_match else header.split('(')[0].strip()
            sections.append({"annotation": annotation,
                              "full": f"## {header}\n{content}"})
            i += 2
        else:
            i += 1
    return [s for s in sections
            if "do not" not in s["annotation"].lower()
            and "disclose" not in s["annotation"].lower()]


class ANDGate:
    def __init__(self, sections):
        self.sections    = sections
        self.model       = SentenceTransformer("all-MiniLM-L6-v2")
        annotations      = [s["annotation"] for s in sections]
        tokenized        = [a.lower().split() for a in annotations]
        self.bm25        = BM25Okapi(tokenized)
        self.ann_embeds  = self.model.encode(annotations, normalize_embeddings=True)

    def score(self, query):
        q_tokens    = query.lower().split()
        bm25_raw    = np.array(self.bm25.get_scores(q_tokens))
        bm25_max    = bm25_raw.max()
        bm25_norm   = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        q_embed     = self.model.encode([query], normalize_embeddings=True)[0]
        cos_scores  = self.ann_embeds @ q_embed

        # Return the per-signal maxima independently
        return float(bm25_norm.max()), float(cos_scores.max())

    def should_inject(self, query, threshold):
        bm25_max, cos_max = self.score(query)
        return bm25_max >= threshold and cos_max >= threshold, bm25_max, cos_max


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


def run_institution(iid):
    name    = INSTITUTIONS[iid]
    v2_text = (DESK / f"burnettbench_{iid}_llmstxt_v2.txt").read_text()
    probes  = [json.loads(l)
               for l in open(DESK / f"burnettbench_{iid}_factprobes_full.jsonl")
               if l.strip()]

    sections = parse_sections(v2_text)
    gate     = ANDGate(sections)

    results_file  = STUDY / f"burnettbench_{iid}_andgate_results.jsonl"
    verdicts_file = STUDY / f"burnettbench_{iid}_andgate_verdicts.jsonl"

    system_a  = "You are a helpful assistant."
    system_v2 = (f"You are a helpful assistant. Use the following reference "
                 f"information if relevant.\n<reference>\n{v2_text}\n</reference>")

    print(f"\n{'─'*65}")
    print(f"{name}: {len(probes)} probes | {len(sections)} sections")

    # Pre-compute AND gate decisions for every probe × threshold
    gate_decisions = {}
    for p in probes:
        bm25_max, cos_max = gate.score(p["probe"])
        gate_decisions[p["fact_id"]] = {"bm25": bm25_max, "cos": cos_max}

    # Show leakage stats per threshold
    for thresh in AND_THRESHOLDS:
        inject_count = sum(
            1 for d in gate_decisions.values()
            if d["bm25"] >= thresh and d["cos"] >= thresh
        )
        print(f"  AND{int(thresh*100)}: {inject_count}/{len(probes)} queries pass gate")

    done_run = load_done(results_file, ["model", "condition", "fact_id"])
    tasks = []

    for mname, mid in MODELS.items():
        for thresh in AND_THRESHOLDS:
            cond = f"AND{int(thresh*100)}"
            for p in probes:
                if (mname, cond, p["fact_id"]) not in done_run:
                    d = gate_decisions[p["fact_id"]]
                    inject = d["bm25"] >= thresh and d["cos"] >= thresh
                    sys_msg = system_v2 if inject else system_a
                    tasks.append((mname, mid, cond, p, sys_msg,
                                  d["bm25"], d["cos"], inject))

    total = len(MODELS) * len(AND_THRESHOLDS) * len(probes)
    print(f"  Runs: {total} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(api_call, mid, sys_msg, p["probe"]): (mname, cond, p, bm25, cos, inj)
                for mname, mid, cond, p, sys_msg, bm25, cos, inj in tasks
            }
            for fut in as_completed(futures):
                mname, cond, p, bm25, cos, inj = futures[fut]
                response = fut.result() or ""
                rec = {"institution": iid, "fact_id": p["fact_id"],
                       "category": p.get("category", ""), "fact": p["fact"],
                       "probe": p["probe"], "model": mname,
                       "condition": cond, "response": response,
                       "in_v2": p.get("in_v2", True),
                       "bm25_score": bm25, "cos_score": cos, "gate_inject": inj}
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
                       "model": r["model"], "condition": r["condition"],
                       "verdict": verdict, "in_v2": r.get("in_v2", True),
                       "bm25_score": r.get("bm25_score", 0),
                       "cos_score": r.get("cos_score", 0),
                       "gate_inject": r.get("gate_inject", True)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
    print("  Judge done ✓")


def print_summary(targets):
    and_conds = [f"AND{int(t*100)}" for t in AND_THRESHOLDS]
    cond_order = ["A", "V2", "OPT2"] + and_conds

    for flag, label in [(False, "OUT_V2 accuracy (goal: >= baseline A)"),
                        (True,  "IN_V2 recall (goal: ~= V2)")]:
        print(f"\n{'='*75}")
        print(label)
        header = f"{'Institution':<22}" + "".join(f"{c:>9}" for c in cond_order)
        print(header)
        print("─" * len(header))

        for iid in targets:
            # Load baseline
            base_v = []
            for fn in [f"burnettbench_{iid}_factprobe_verdicts_full.jsonl",
                       f"burnettbench_{iid}_gating_verdicts.jsonl"]:
                fp = STUDY / fn
                if not fp.exists(): fp = DESK / fn
                if fp.exists():
                    base_v += [json.loads(l) for l in open(fp) if l.strip()]

            and_v = []
            af = STUDY / f"burnettbench_{iid}_andgate_verdicts.jsonl"
            if af.exists():
                and_v = [json.loads(l) for l in open(af) if l.strip()]

            all_v  = base_v + and_v
            subset = [v for v in all_v if v.get("in_v2") == flag]
            row    = f"{INSTITUTIONS[iid]:<22}"
            for cond in cond_order:
                recs = [v for v in subset if v["condition"] == cond]
                if recs:
                    acc = sum(1 for v in recs if v["verdict"] == "A") / len(recs) * 100
                    row += f"{acc:>8.1f}%"
                else:
                    row += f"{'—':>9}"
            print(row)

        # Pooled mean
        row = f"{'MEAN':<22}"
        for cond in cond_order:
            accs = []
            for iid in targets:
                base_v = []
                for fn in [f"burnettbench_{iid}_factprobe_verdicts_full.jsonl",
                           f"burnettbench_{iid}_gating_verdicts.jsonl"]:
                    fp = STUDY / fn
                    if not fp.exists(): fp = DESK / fn
                    if fp.exists():
                        base_v += [json.loads(l) for l in open(fp) if l.strip()]
                and_v = []
                af = STUDY / f"burnettbench_{iid}_andgate_verdicts.jsonl"
                if af.exists():
                    and_v = [json.loads(l) for l in open(af) if l.strip()]
                all_v  = base_v + and_v
                subset = [v for v in all_v if v.get("in_v2") == flag]
                recs   = [v for v in subset if v["condition"] == cond]
                if recs:
                    accs.append(sum(1 for v in recs if v["verdict"] == "A") / len(recs) * 100)
            row += f"{sum(accs)/len(accs):>8.1f}%" if accs else f"{'—':>9}"
        print("─" * len(header))
        print(row)


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    targets = [t for t in targets if t in INSTITUTIONS]

    print(f"BurnettBench — AND Gate | thresholds: {[f'AND{int(t*100)}' for t in AND_THRESHOLDS]}")
    for iid in targets:
        run_institution(iid)
    print_summary(targets)


if __name__ == "__main__":
    main()
