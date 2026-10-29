#!/usr/bin/env python3
"""
BurnettBench — OPT3 k-Sweep Study
Tests Section-Level Retrieval at k = 1, 2, 3, 5, full-doc across all 5 institutions.

Research question: Does IN_V2 recall recover as k increases?
Is there a k sweet-spot where both IN_V2 recall and OUT_V2 accuracy are high?

Uses existing factprobe questions and llms.txt V2 documents.
Outputs: burnettbench_{iid}_ksweep_verdicts.jsonl

Usage: python burnettbench_eval_ksweep.py [institution_id]
"""
import sys, json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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

# k values to test; "full" = inject entire V2 document
K_VALUES = [1, 2, 3, 5, "full"]

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


# ── Section parsing ────────────────────────────────────────────────────────────

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
    return sections


# ── Retriever ──────────────────────────────────────────────────────────────────

class SectionRetriever:
    def __init__(self, sections):
        self.sections    = sections
        self.model       = SentenceTransformer("all-MiniLM-L6-v2")
        annotations      = [s["annotation"] for s in sections]
        tokenized        = [a.lower().split() for a in annotations]
        self.bm25        = BM25Okapi(tokenized)
        self.ann_embeds  = self.model.encode(annotations, normalize_embeddings=True)

    def top_k(self, query, k):
        q_tokens    = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        bm25_max    = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
        q_embed     = self.model.encode([query], normalize_embeddings=True)[0]
        cos_scores  = self.ann_embeds @ q_embed
        hybrid      = 0.5 * bm25_scores + 0.5 * cos_scores
        top_idx     = np.argsort(hybrid)[::-1][:k]
        return "\n\n".join(self.sections[i]["full"] for i in top_idx)


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


# ── Per-institution runner ─────────────────────────────────────────────────────

def run_institution(iid):
    name    = INSTITUTIONS[iid]
    v2_text = (STUDY / f"burnettbench_{iid}_llmstxt_v2.txt").read_text()
    probes  = [json.loads(l)
               for l in open(STUDY / f"burnettbench_{iid}_factprobes_full.jsonl")
               if l.strip()]

    sections  = parse_sections(v2_text)
    retriever = SectionRetriever(sections)
    n_secs    = len(sections)

    # Resolve "full" → actual section count; cap k at n_secs
    effective_k = {k: (n_secs if k == "full" else min(k, n_secs)) for k in K_VALUES}

    verdicts_file = STUDY / f"burnettbench_{iid}_ksweep_verdicts.jsonl"
    results_file  = STUDY / f"burnettbench_{iid}_ksweep_results.jsonl"

    in_v2_count  = sum(1 for p in probes if p.get("in_v2", True))
    out_v2_count = len(probes) - in_v2_count

    print(f"\n{'─'*65}")
    print(f"{name}: {len(probes)} probes ({in_v2_count} IN_V2, {out_v2_count} OUT_V2)")
    print(f"  Sections: {n_secs} | k sweep: {[effective_k[k] for k in K_VALUES]}")

    # Pre-compute top-k context for each probe × k combination
    contexts = {}  # (fact_id, k_label) -> context_string
    for p in probes:
        for k in K_VALUES:
            ek = effective_k[k]
            label = str(k)
            contexts[(p["fact_id"], label)] = retriever.top_k(p["probe"], ek)

    full_v2_context = v2_text

    # ── Step 1: Run models ─────────────────────────────────────────────────────
    done_run = load_done(results_file, ["model", "condition", "fact_id"])

    tasks = []
    for mname, mid in MODELS.items():
        for k in K_VALUES:
            label = str(k)
            cond  = f"OPT3_k{label}"
            for p in probes:
                if (mname, cond, p["fact_id"]) not in done_run:
                    ctx = (full_v2_context if k == "full"
                           else contexts[(p["fact_id"], label)])
                    sys_msg = (f"You are a helpful assistant. Use the following "
                               f"reference information if relevant.\n<reference>\n"
                               f"{ctx}\n</reference>")
                    tasks.append((mname, mid, cond, p, sys_msg))

    total_planned = len(MODELS) * len(K_VALUES) * len(probes)
    print(f"  Runs: {total_planned} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {
                ex.submit(api_call, mid, sys_msg, p["probe"]): (mname, cond, p)
                for mname, mid, cond, p, sys_msg in tasks
            }
            for fut in as_completed(futures):
                mname, cond, p = futures[fut]
                response = fut.result() or ""
                rec = {"institution": iid, "fact_id": p["fact_id"],
                       "category": p.get("category", ""), "fact": p["fact"],
                       "probe": p["probe"], "model": mname,
                       "condition": cond, "response": response,
                       "in_v2": p.get("in_v2", True)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 50 == 0:
                    print(f"    {completed}/{len(tasks)} responses...", flush=True)
    print(f"  Runner done ✓")

    # ── Step 2: Judge ─────────────────────────────────────────────────────────
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
                       "verdict": verdict, "in_v2": r.get("in_v2", True)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 50 == 0:
                    print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
    print(f"  Judge done ✓")


# ── Summary (cross-institution) ────────────────────────────────────────────────

def print_summary(targets):
    # Load baseline V2 and A verdicts for comparison
    baseline = {}  # iid -> list of verdict dicts
    for iid in targets:
        vf = STUDY / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
        baseline[iid] = [json.loads(l) for l in open(vf) if l.strip()] if vf.exists() else []

    k_labels = [str(k) for k in K_VALUES]
    cond_order = ["A", "V2"] + [f"OPT3_k{k}" for k in k_labels]

    print(f"\n{'='*70}")
    for flag, label in [(True, "IN_V2 recall (higher = better)"),
                        (False, "OUT_V2 accuracy (higher = better)")]:
        print(f"\n{label}")
        header = f"{'Institution':<22}" + "".join(f"{c:>10}" for c in cond_order)
        print(header)
        print("─" * len(header))
        for iid in targets:
            all_v = baseline[iid].copy()
            ksweep_file = STUDY / f"burnettbench_{iid}_ksweep_verdicts.jsonl"
            if ksweep_file.exists():
                all_v += [json.loads(l) for l in open(ksweep_file) if l.strip()]
            subset = [v for v in all_v if v.get("in_v2") == flag]
            row = f"{INSTITUTIONS[iid]:<22}"
            for cond in cond_order:
                recs = [v for v in subset if v["condition"] == cond]
                acc  = (sum(1 for v in recs if v["verdict"] == "A") / len(recs) * 100
                        if recs else float("nan"))
                row += f"{acc:>9.1f}%" if recs else f"{'—':>10}"
            print(row)

        # Grand mean row
        row = f"{'MEAN':<22}"
        for cond in cond_order:
            accs = []
            for iid in targets:
                all_v = baseline[iid].copy()
                ksweep_file = STUDY / f"burnettbench_{iid}_ksweep_verdicts.jsonl"
                if ksweep_file.exists():
                    all_v += [json.loads(l) for l in open(ksweep_file) if l.strip()]
                subset = [v for v in all_v if v.get("in_v2") == flag]
                recs   = [v for v in subset if v["condition"] == cond]
                if recs:
                    accs.append(sum(1 for v in recs if v["verdict"] == "A") / len(recs) * 100)
            row += f"{sum(accs)/len(accs):>9.1f}%" if accs else f"{'—':>10}"
        print("─" * len(header))
        print(row)


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    targets = [t for t in targets if t in INSTITUTIONS]

    print(f"BurnettBench — OPT3 k-Sweep | institutions: {', '.join(targets)}")
    print(f"k values: {K_VALUES}")

    for iid in targets:
        run_institution(iid)

    print_summary(targets)


if __name__ == "__main__":
    main()
