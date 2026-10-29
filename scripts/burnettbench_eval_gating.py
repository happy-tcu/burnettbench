#!/usr/bin/env python3
"""
BurnettBench — Gating Experiment (runs on DGX)
Tests Options 2 and 3 for eliminating out-of-scope accuracy harm.

Option 2 — Scope-aware injection (simulates llms.txt spec extension):
  Uses (use for: ...) annotations in V2 section headers as a scope
  declaration. For each query, BM25+embedding hybrid scores the query
  against section annotations. If max section score < threshold → don't
  inject V2. Simulates what a covers/excludes metadata field would enable.

Option 3 — Section-level retrieval:
  Instead of injecting full V2, embed query against each section,
  inject only top-k matching sections. Model never sees irrelevant sections.

Both run via OpenRouter (same 5 models). Gating/retrieval runs locally on DGX.

Usage: python burnettbench_eval_gating.py [institution_id]
"""
import sys, json, re, time, math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DESK   = Path("/home/happy/burnettbench")
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
    "qwen3-14b":       "qwen/qwen3-14b",
    "gpt-4o-mini":     "openai/gpt-4o-mini",
    "llama-3.1-70b":   "meta-llama/llama-3.1-70b-instruct",
    "claude-haiku-4-5":"anthropic/claude-haiku-4-5",
    "gemini-flash":    "google/gemini-3.8-flash",
}

# Gating threshold — tune if needed
GATE_THRESHOLD = 0.35
# Sections to inject for Option 3
TOP_K_SECTIONS = 2

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
    """Split V2 into sections. Returns list of (header, content, annotation)."""
    sections = []
    parts = re.split(r'\n(##[^#].*)', v2_text)
    # parts: [preamble, header1, content1, header2, content2, ...]
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if chunk.startswith('##'):
            header  = chunk.lstrip('#').strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            # Extract (use for: ...) annotation if present
            ann_match = re.search(r'\(use for:\s*(.+?)\)', header, re.IGNORECASE)
            annotation = ann_match.group(1) if ann_match else header.split('(')[0].strip()
            clean_header = header.split('(')[0].strip()
            sections.append({
                "header":     clean_header,
                "annotation": annotation,
                "content":    content,
                "full":       f"## {header}\n{content}"
            })
            i += 2
        else:
            i += 1
    return [s for s in sections if "do not" not in s["header"].lower()
            and "disclose" not in s["header"].lower()]


# ── Gating classifier ──────────────────────────────────────────────────────────

class GatingClassifier:
    def __init__(self, sections):
        self.sections   = sections
        self.model      = SentenceTransformer("all-MiniLM-L6-v2")
        annotations     = [s["annotation"] for s in sections]
        # BM25 on annotation tokens
        tokenized       = [a.lower().split() for a in annotations]
        self.bm25       = BM25Okapi(tokenized)
        # Dense embeddings of annotations
        self.ann_embeds = self.model.encode(annotations, normalize_embeddings=True)

    def score(self, query):
        """Returns (max_score, section_scores) hybrid BM25+embedding."""
        q_tokens  = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        # Normalize BM25 to [0,1]
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        q_embed   = self.model.encode([query], normalize_embeddings=True)[0]
        cos_scores = self.ann_embeds @ q_embed  # cosine similarity

        # Hybrid: equal weight
        hybrid = 0.5 * bm25_scores + 0.5 * cos_scores
        return hybrid.max(), hybrid

    def should_inject(self, query, threshold=GATE_THRESHOLD):
        """Option 2: inject full V2 only if max section score >= threshold."""
        max_score, _ = self.score(query)
        return max_score >= threshold, max_score

    def top_sections(self, query, k=TOP_K_SECTIONS):
        """Option 3: return top-k section content strings."""
        _, scores = self.score(query)
        top_idx   = np.argsort(scores)[::-1][:k]
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


# ── Main ───────────────────────────────────────────────────────────────────────

def run_institution(iid):
    name     = INSTITUTIONS[iid]
    v2_text  = open(DESK / f"burnettbench_{iid}_llmstxt_v2.txt").read()
    probes   = [json.loads(l) for l in
                open(DESK / f"burnettbench_{iid}_factprobes_full.jsonl") if l.strip()]

    sections  = parse_sections(v2_text)
    gate      = GatingClassifier(sections)

    results_file  = DESK / f"burnettbench_{iid}_gating_results.jsonl"
    verdicts_file = DESK / f"burnettbench_{iid}_gating_verdicts.jsonl"

    system_v2 = (f"You are a helpful assistant. Use the following reference "
                 f"information if relevant.\n<reference>\n{v2_text}\n</reference>")
    system_a  = "You are a helpful assistant."

    print(f"\n{'─'*65}")
    print(f"{name}: {len(probes)} probes | {len(sections)} V2 sections")
    print(f"  Sections: {[s['header'] for s in sections]}")

    # Pre-compute gating decisions for all probes
    gate_decisions = {}
    for p in probes:
        inject, score = gate.should_inject(p["probe"])
        top_secs      = gate.top_sections(p["probe"])
        gate_decisions[p["fact_id"]] = {
            "inject": bool(inject), "score": float(score), "top_sections": top_secs
        }

    in_scope  = sum(1 for d in gate_decisions.values() if d["inject"])
    out_scope = sum(1 for d in gate_decisions.values() if not d["inject"])
    print(f"  Gate decisions: {in_scope} inject | {out_scope} skip")

    # Show gate decisions by in_v2 flag
    for p in probes[:3]:
        d = gate_decisions[p["fact_id"]]
        print(f"    [{p.get('in_v2','?')}] score={d['score']:.3f} inject={d['inject']} | {p['probe'][:60]}")

    # ── Step 1: Run models ─────────────────────────────────────────────────────
    done_run = load_done(results_file, ["model", "condition", "fact_id"])
    tasks = []
    for mname, mid in MODELS.items():
        for p in probes:
            d = gate_decisions[p["fact_id"]]
            # OPT2: inject full V2 only if gate says inject, else no context
            for cond in ["OPT2", "OPT3"]:
                if (mname, cond, p["fact_id"]) not in done_run:
                    tasks.append((mname, mid, cond, p, d))

    total = len(MODELS) * 2 * len(probes)
    print(f"  Runs: {total} total | {len(done_run)} done | {len(tasks)} remaining")

    completed = 0
    with open(results_file, "a") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {}
            for mname, mid, cond, p, d in tasks:
                if cond == "OPT2":
                    sys = system_v2 if d["inject"] else system_a
                else:  # OPT3
                    if d["top_sections"]:
                        sys = (f"You are a helpful assistant. Use the following "
                               f"reference information if relevant.\n<reference>\n"
                               f"{d['top_sections']}\n</reference>")
                    else:
                        sys = system_a
                futures[ex.submit(api_call, mid, sys, p["probe"])] = (mname, cond, p, d)

            for fut in as_completed(futures):
                mname, cond, p, d = futures[fut]
                response = fut.result() or ""
                rec = {"institution": iid, "fact_id": p["fact_id"],
                       "category": p["category"], "fact": p["fact"],
                       "probe": p["probe"], "model": mname,
                       "condition": cond, "response": response,
                       "in_v2": p.get("in_v2", True),
                       "gate_score": float(d["score"]),
                       "gate_inject": d["inject"]}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                completed += 1
                if completed % 100 == 0:
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
    print(f"  Judge done ✓")

    # ── Summary ────────────────────────────────────────────────────────────────
    orig_file = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
    orig      = [json.loads(l) for l in open(orig_file) if l.strip()] if orig_file.exists() else []
    gate_v    = [json.loads(l) for l in open(verdicts_file) if l.strip()]
    all_v     = orig + gate_v

    print(f"\n  {'─'*55}")
    for label, flag in [("IN_V2", True), ("OUT_V2", False)]:
        subset = [v for v in all_v if v.get("in_v2") == flag]
        if not subset: continue
        n_facts = len(set(v["fact_id"] for v in subset))
        print(f"\n  {label} (n={n_facts} facts)")
        print(f"  {'Cond':<8}{'Mean acc':>10}  {'vs V2':>8}")
        cond_means = {}
        for cond in ["A", "V2", "OPT2", "OPT3"]:
            recs = [v for v in subset if v["condition"] == cond]
            if not recs: continue
            acc = sum(1 for v in recs if v["verdict"] == "A") / len(recs) * 100
            cond_means[cond] = acc
            delta = acc - cond_means.get("V2", acc)
            marker = " ← TARGET" if cond in ["OPT2","OPT3"] and flag == False else ""
            print(f"  {cond:<8}{acc:>9.1f}%  {delta:>+7.1f}pp{marker}")


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    print("GATING EXPERIMENT — Options 2 & 3")
    print("OPT2: scope-aware injection (gate on/off)")
    print("OPT3: section-level retrieval (top-2 sections only)")
    for iid in [t for t in targets if t in INSTITUTIONS]:
        run_institution(iid)

    # Grand summary
    if len(targets) > 1:
        print(f"\n{'='*65}")
        print("GRAND SUMMARY — OUT_V2 accuracy")
        print(f"{'Institution':<22}{'A':>8}{'V2':>8}{'OPT2':>8}{'OPT3':>8}")
        for iid in targets:
            vf_o = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
            vf_g = DESK / f"burnettbench_{iid}_gating_verdicts.jsonl"
            if not vf_o.exists() or not vf_g.exists(): continue
            all_v = ([json.loads(l) for l in open(vf_o) if l.strip()] +
                     [json.loads(l) for l in open(vf_g) if l.strip()])
            out   = [v for v in all_v if not v.get("in_v2")]
            row   = f"{INSTITUTIONS[iid]:<22}"
            for cond in ["A","V2","OPT2","OPT3"]:
                recs = [v for v in out if v["condition"]==cond]
                acc  = sum(1 for v in recs if v["verdict"]=="A")/len(recs)*100 if recs else 0.0
                row += f"{acc:>7.1f}%"
            print(row)

        print(f"\n{'='*65}")
        print("IN_V2 recall — must stay high")
        print(f"{'Institution':<22}{'A':>8}{'V2':>8}{'OPT2':>8}{'OPT3':>8}")
        for iid in targets:
            vf_o = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
            vf_g = DESK / f"burnettbench_{iid}_gating_verdicts.jsonl"
            if not vf_o.exists() or not vf_g.exists(): continue
            all_v = ([json.loads(l) for l in open(vf_o) if l.strip()] +
                     [json.loads(l) for l in open(vf_g) if l.strip()])
            inv   = [v for v in all_v if v.get("in_v2")]
            row   = f"{INSTITUTIONS[iid]:<22}"
            for cond in ["A","V2","OPT2","OPT3"]:
                recs = [v for v in inv if v["condition"]==cond]
                acc  = sum(1 for v in recs if v["verdict"]=="A")/len(recs)*100 if recs else 0.0
                row += f"{acc:>7.1f}%"
            print(row)


if __name__ == "__main__":
    main()
