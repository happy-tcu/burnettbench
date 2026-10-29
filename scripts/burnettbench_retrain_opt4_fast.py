#!/usr/bin/env python3
"""
BurnettBench — OPT4v2 Fast Retrain
Logistic regression on SentenceTransformer embeddings.
Trains in seconds per fold vs minutes for DistilBERT fine-tuning.

Usage: python burnettbench_retrain_opt4_fast.py
"""
import json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

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


def parse_annotations(v2_text):
    annotations = []
    parts = re.split(r'\n(##[^#].*)', v2_text)
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if chunk.startswith('##'):
            header    = chunk.lstrip('#').strip()
            ann_match = re.search(r'\(use for:\s*(.+?)\)', header, re.IGNORECASE)
            ann = ann_match.group(1) if ann_match else header.split('(')[0].strip()
            if "do not" not in ann.lower() and "disclose" not in ann.lower():
                annotations.append(ann)
            i += 2
        else:
            i += 1
    return " | ".join(annotations)


def make_input(query, annotations):
    """Concatenate query + annotations as single string for embedding."""
    return f"{query} [SEP] {annotations}"


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
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading SentenceTransformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    synthetic = [json.loads(l)
                 for l in open(STUDY / "burnettbench_opt4_synthetic_all.jsonl")
                 if l.strip()]

    all_data = {}
    for iid in INSTITUTIONS:
        v2_text     = (DESK / f"burnettbench_{iid}_llmstxt_v2.txt").read_text()
        annotations = parse_annotations(v2_text)
        probes      = [json.loads(l)
                       for l in open(DESK / f"burnettbench_{iid}_factprobes_full.jsonl")
                       if l.strip()]
        gt_examples  = [(p["probe"], annotations, int(p.get("in_v2", True)))
                        for p in probes]
        syn_examples = [(r["question"], annotations, r["label"])
                        for r in synthetic if r["institution"] == iid]
        all_examples = gt_examples + syn_examples

        all_data[iid] = {
            "probes":       probes,
            "annotations":  annotations,
            "v2_text":      v2_text,
            "all_examples": all_examples,
        }
        print(f"{iid}: {len(gt_examples)} GT + {len(syn_examples)} syn = {len(all_examples)}")

    # ── Pre-encode all texts ───────────────────────────────────────────────────
    print("\nEncoding all examples...")
    all_texts  = []
    all_labels = []
    all_iids   = []
    for iid, data in all_data.items():
        for q, ann, label in data["all_examples"]:
            all_texts.append(make_input(q, ann))
            all_labels.append(label)
            all_iids.append(iid)

    embeddings = embedder.encode(all_texts, batch_size=64,
                                 show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoded {len(embeddings)} examples")

    # ── Leave-one-out CV ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("LEAVE-ONE-OUT CROSS-VALIDATION")

    cv_results  = {}
    classifiers = {}

    for holdout in INSTITUTIONS:
        print(f"\n── Holdout: {INSTITUTIONS[holdout]} ──")
        train_idx = [i for i, iid in enumerate(all_iids) if iid != holdout]
        val_idx   = [i for i, iid in enumerate(all_iids) if iid == holdout]

        X_train = embeddings[train_idx]
        y_train = [all_labels[i] for i in train_idx]
        X_val   = embeddings[val_idx]
        y_val   = [all_labels[i] for i in val_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc   = sum(p == l for p, l in zip(preds, y_val)) / len(y_val)
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Acc: {acc:.3f}")
        print(classification_report(y_val, preds, target_names=["OUT_V2","IN_V2"]))

        cv_results[holdout]  = list(zip(y_val, preds.tolist()))
        classifiers[holdout] = clf

        # Predict on original GT probes only
        gt_preds = {}
        for p in all_data[holdout]["probes"]:
            text = make_input(p["probe"], all_data[holdout]["annotations"])
            emb  = embedder.encode([text], normalize_embeddings=True)
            pred = int(clf.predict(emb)[0])
            gt_preds[p["fact_id"]] = pred
        all_data[holdout]["opt4v2_preds"] = gt_preds

    # Overall
    all_true = [l for r in cv_results.values() for l, _ in r]
    all_pred = [p for r in cv_results.values() for _, p in r]
    overall  = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    print(f"\n{'='*65}")
    print(f"Overall CV accuracy: {overall:.3f}")
    print(classification_report(all_true, all_pred, target_names=["OUT_V2","IN_V2"]))

    print(f"\n{'Institution':<22} {'IN_V2 passed':>14} {'OUT_V2 blocked':>16}")
    for iid in INSTITUTIONS:
        preds  = all_data[iid].get("opt4v2_preds", {})
        probes = all_data[iid]["probes"]
        in_v2  = [p for p in probes if p.get("in_v2", True)]
        out_v2 = [p for p in probes if not p.get("in_v2", True)]
        in_pass   = sum(1 for p in in_v2  if preds.get(p["fact_id"], 1) == 1)
        out_block = sum(1 for p in out_v2 if preds.get(p["fact_id"], 1) == 0)
        print(f"  {INSTITUTIONS[iid]:<20} {in_pass}/{len(in_v2)} "
              f"({in_pass/len(in_v2)*100:.0f}%)       "
              f"{out_block}/{len(out_v2)} ({out_block/len(out_v2)*100:.0f}%)")

    # ── Benchmark ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RUNNING OPT4v2 BENCHMARK")

    for iid in INSTITUTIONS:
        name    = INSTITUTIONS[iid]
        data    = all_data[iid]
        probes  = data["probes"]
        preds   = data.get("opt4v2_preds", {})
        v2_text = data["v2_text"]

        results_file  = STUDY / f"burnettbench_{iid}_opt4v2_results.jsonl"
        verdicts_file = STUDY / f"burnettbench_{iid}_opt4v2_verdicts.jsonl"

        system_a  = "You are a helpful assistant."
        system_v2 = (f"You are a helpful assistant. Use the following reference "
                     f"information if relevant.\n<reference>\n{v2_text}\n</reference>")

        done_run = load_done(results_file, ["model", "condition", "fact_id"])
        tasks = []
        for mname, mid in MODELS.items():
            for p in probes:
                if (mname, "OPT4v2", p["fact_id"]) not in done_run:
                    inject  = preds.get(p["fact_id"], 1) == 1
                    sys_msg = system_v2 if inject else system_a
                    tasks.append((mname, mid, p, sys_msg, inject))

        print(f"\n{name}: {len(tasks)} remaining runs")

        completed = 0
        with open(results_file, "a") as f:
            with ThreadPoolExecutor(max_workers=20) as ex:
                futures = {
                    ex.submit(api_call, mid, sys_msg, p["probe"]): (mname, p, inj)
                    for mname, mid, p, sys_msg, inj in tasks
                }
                for fut in as_completed(futures):
                    mname, p, inj = futures[fut]
                    response = fut.result() or ""
                    rec = {"institution": iid, "fact_id": p["fact_id"],
                           "fact": p["fact"], "probe": p["probe"],
                           "model": mname, "condition": "OPT4v2",
                           "response": response, "in_v2": p.get("in_v2", True),
                           "gate_inject": inj}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"    {completed}/{len(tasks)} responses...", flush=True)
        print("  Runner done ✓")

        responses   = [json.loads(l) for l in open(results_file) if l.strip()
                       and not json.loads(l)["response"].startswith("ERROR")]
        done_judge  = load_done(verdicts_file, ["model", "condition", "fact_id"])
        judge_tasks = [r for r in responses
                       if (r["model"], r["condition"], r["fact_id"]) not in done_judge]

        completed = 0
        with open(verdicts_file, "a") as f:
            with ThreadPoolExecutor(max_workers=15) as ex:
                fmap = {
                    ex.submit(judge_call,
                        JUDGE_PROMPT.format(institution=name, fact=r["fact"],
                                            question=r["probe"],
                                            response=r["response"])): r
                    for r in judge_tasks
                }
                for fut in as_completed(fmap):
                    r = fmap[fut]
                    verdict = (fut.result() or "")[:1].upper()
                    rec = {"institution": iid, "fact_id": r["fact_id"],
                           "model": r["model"], "condition": r["condition"],
                           "verdict": verdict, "in_v2": r.get("in_v2", True),
                           "gate_inject": r.get("gate_inject", True)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
        print("  Judge done ✓")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL RESULTS — OPT4v2 vs OPT4 vs OPT2 vs BASELINES")
    cond_order = ["A", "V2", "OPT2", "OPT4", "OPT4v2"]

    for flag, label in [(False, "OUT_V2 accuracy (goal: >= A)"),
                        (True,  "IN_V2 recall   (goal: ~= V2)")]:
        print(f"\n{label}")
        print(f"{'Institution':<22}" + "".join(f"{c:>9}" for c in cond_order))
        print("─" * (22 + 9 * len(cond_order)))

        for iid in INSTITUTIONS:
            all_v = []
            for fn in [f"burnettbench_{iid}_factprobe_verdicts_full.jsonl",
                       f"burnettbench_{iid}_gating_verdicts.jsonl",
                       f"burnettbench_{iid}_opt4_verdicts.jsonl",
                       f"burnettbench_{iid}_opt4v2_verdicts.jsonl"]:
                fp = STUDY / fn
                if not fp.exists(): fp = DESK / fn
                if fp.exists():
                    all_v += [json.loads(l) for l in open(fp) if l.strip()]

            subset = [v for v in all_v if v.get("in_v2") == flag]
            row    = f"{INSTITUTIONS[iid]:<22}"
            for cond in cond_order:
                recs = [v for v in subset if v["condition"] == cond]
                row += (f"{sum(1 for v in recs if v['verdict']=='A')/len(recs)*100:>8.1f}%"
                        if recs else f"{'—':>9}")
            print(row)

        row = f"{'MEAN':<22}"
        for cond in cond_order:
            accs = []
            for iid in INSTITUTIONS:
                all_v = []
                for fn in [f"burnettbench_{iid}_factprobe_verdicts_full.jsonl",
                           f"burnettbench_{iid}_gating_verdicts.jsonl",
                           f"burnettbench_{iid}_opt4_verdicts.jsonl",
                           f"burnettbench_{iid}_opt4v2_verdicts.jsonl"]:
                    fp = STUDY / fn
                    if not fp.exists(): fp = DESK / fn
                    if fp.exists():
                        all_v += [json.loads(l) for l in open(fp) if l.strip()]
                subset = [v for v in all_v if v.get("in_v2") == flag]
                recs   = [v for v in subset if v["condition"] == cond]
                if recs:
                    accs.append(sum(1 for v in recs if v["verdict"]=="A")/len(recs)*100)
            row += f"{sum(accs)/len(accs):>8.1f}%" if accs else f"{'—':>9}"
        print("─" * (22 + 9 * len(cond_order)))
        print(row)


if __name__ == "__main__":
    main()
