#!/usr/bin/env python3
"""
BurnettBench — OPT5 and OPT4v3
Direction 1: OPT5   = LR on clean GT only (no synthetic)
Direction 3: OPT4v3 = LR on GT + confidence-filtered synthetic

Denoising (per fold): train GT-only LR, score each synthetic example,
keep only those where prediction agrees with LLM label AND confidence >= threshold.
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

DENOISE_THRESHOLDS = [0.60, 0.70, 0.80]
RUN_BENCHMARK = True


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


def run_loo_cv(variant_label, gt_embs, gt_labels, gt_iids,
               syn_embs, syn_labels, syn_iids,
               denoise_threshold=None):
    """
    Leave-one-out CV. Returns (cv_results, per_holdout_gate_preds).
    denoise_threshold=None → OPT5 (GT only)
    denoise_threshold=float → OPT4v3 (GT + filtered synthetic)
    """
    print(f"\n{'─'*65}")
    print(f"CV: {variant_label}")
    cv_results = {}
    gate_preds = {}   # holdout → {fact_id: 0/1}

    for holdout in INSTITUTIONS:
        gt_train_mask = [iid != holdout for iid in gt_iids]
        gt_val_mask   = [iid == holdout for iid in gt_iids]

        X_gt_tr = gt_embs[[i for i, m in enumerate(gt_train_mask) if m]]
        y_gt_tr = [l for l, m in zip(gt_labels, gt_train_mask) if m]
        X_val   = gt_embs[[i for i, m in enumerate(gt_val_mask)   if m]]
        y_val   = [l for l, m in zip(gt_labels, gt_val_mask)   if m]

        if denoise_threshold is not None and syn_embs is not None:
            syn_train_mask = [iid != holdout for iid in syn_iids]
            X_syn_cand = syn_embs[[i for i, m in enumerate(syn_train_mask) if m]]
            y_syn_cand = [l for l, m in zip(syn_labels, syn_train_mask) if m]

            # Train GT-only classifier to score synthetics
            clf_gt = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf_gt.fit(X_gt_tr, y_gt_tr)

            proba = clf_gt.predict_proba(X_syn_cand)
            preds = clf_gt.predict(X_syn_cand)
            conf  = proba.max(axis=1)

            keep = [i for i in range(len(y_syn_cand))
                    if conf[i] >= denoise_threshold and preds[i] == y_syn_cand[i]]

            if keep:
                X_train = np.vstack([X_gt_tr, X_syn_cand[keep]])
                y_train = y_gt_tr + [y_syn_cand[i] for i in keep]
            else:
                X_train, y_train = X_gt_tr, y_gt_tr

            n_kept = len(keep)
        else:
            X_train, y_train = X_gt_tr, y_gt_tr
            n_kept = 0

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)

        val_preds = clf.predict(X_val)
        acc = sum(p == l for p, l in zip(val_preds, y_val)) / len(y_val)
        print(f"  Holdout {INSTITUTIONS[holdout]:<22} "
              f"GT: {len(y_gt_tr):>3} + Syn: {n_kept:>3} | "
              f"Val: {len(y_val):>3} | Acc: {acc:.3f}")

        cv_results[holdout] = list(zip(y_val, val_preds.tolist()))
        gate_preds[holdout] = clf  # store clf, encode probes later

    all_true = [l for r in cv_results.values() for l, _ in r]
    all_pred = [p for r in cv_results.values() for _, p in r]
    overall  = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    print(f"  Overall CV accuracy: {overall:.3f}")
    print(classification_report(all_true, all_pred, target_names=["OUT_V2","IN_V2"]))

    return cv_results, gate_preds


def run_benchmark(variant_label, condition_name, all_data, embedder,
                  gt_embs, gt_labels, gt_iids, syn_embs, syn_labels, syn_iids,
                  denoise_threshold=None):
    """Run full benchmark for a variant, saving results and verdicts."""
    print(f"\n{'='*65}")
    print(f"RUNNING BENCHMARK: {variant_label}")

    # Rerun CV to get per-holdout classifiers (same logic as run_loo_cv)
    holdout_clfs = {}
    for holdout in INSTITUTIONS:
        gt_train_mask = [iid != holdout for iid in gt_iids]
        X_gt_tr = gt_embs[[i for i, m in enumerate(gt_train_mask) if m]]
        y_gt_tr = [l for l, m in zip(gt_labels, gt_train_mask) if m]

        if denoise_threshold is not None and syn_embs is not None:
            syn_train_mask = [iid != holdout for iid in syn_iids]
            X_syn_cand = syn_embs[[i for i, m in enumerate(syn_train_mask) if m]]
            y_syn_cand = [l for l, m in zip(syn_labels, syn_train_mask) if m]

            clf_gt = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf_gt.fit(X_gt_tr, y_gt_tr)
            proba = clf_gt.predict_proba(X_syn_cand)
            preds = clf_gt.predict(X_syn_cand)
            conf  = proba.max(axis=1)
            keep = [i for i in range(len(y_syn_cand))
                    if conf[i] >= denoise_threshold and preds[i] == y_syn_cand[i]]
            if keep:
                X_train = np.vstack([X_gt_tr, X_syn_cand[keep]])
                y_train = y_gt_tr + [y_syn_cand[i] for i in keep]
            else:
                X_train, y_train = X_gt_tr, y_gt_tr
        else:
            X_train, y_train = X_gt_tr, y_gt_tr

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)
        holdout_clfs[holdout] = clf

    for iid in INSTITUTIONS:
        name    = INSTITUTIONS[iid]
        data    = all_data[iid]
        probes  = data["probes"]
        v2_text = data["v2_text"]
        clf     = holdout_clfs[iid]

        # Gate predictions for each probe
        opt_preds = {}
        for p in probes:
            text = make_input(p["probe"], data["annotations"])
            emb  = embedder.encode([text], normalize_embeddings=True)
            pred = int(clf.predict(emb)[0])
            opt_preds[p["fact_id"]] = pred
        data[f"{condition_name}_preds"] = opt_preds

        results_file  = STUDY / f"burnettbench_{iid}_{condition_name}_results.jsonl"
        verdicts_file = STUDY / f"burnettbench_{iid}_{condition_name}_verdicts.jsonl"

        system_a  = "You are a helpful assistant."
        system_v2 = (f"You are a helpful assistant. Use the following reference "
                     f"information if relevant.\n<reference>\n{v2_text}\n</reference>")

        done_run = load_done(results_file, ["model", "condition", "fact_id"])
        tasks = []
        for mname, mid in MODELS.items():
            for p in probes:
                if (mname, condition_name, p["fact_id"]) not in done_run:
                    inject  = opt_preds.get(p["fact_id"], 1) == 1
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
                           "model": mname, "condition": condition_name,
                           "response": response, "in_v2": p.get("in_v2", True),
                           "gate_inject": inj}
                    f.write(json.dumps(rec) + "\n"); f.flush()
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
                           "model": r["model"], "condition": condition_name,
                           "verdict": verdict, "in_v2": r.get("in_v2", True),
                           "gate_inject": r.get("gate_inject", True)}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
        print("  Judge done ✓")


def print_summary(conditions):
    """Print final results table across all conditions."""
    print(f"\n{'='*65}")
    print("FINAL RESULTS COMPARISON")

    all_cond_labels = ["A", "V2", "OPT2", "OPT4", "OPT4v2"] + [c for c in conditions if c not in ("A","V2","OPT2","OPT4","OPT4v2")]

    verdict_files = {
        "A":      "factprobe_verdicts_full",
        "V2":     "factprobe_verdicts_full",
        "OPT2":   "gating_verdicts",
        "OPT4":   "opt4_verdicts",
        "OPT4v2": "opt4v2_verdicts",
    }
    for c in conditions:
        if c not in verdict_files:
            verdict_files[c] = f"{c}_verdicts"

    def load_verdicts(iid):
        all_v = []
        seen = set()
        for cond, suffix in verdict_files.items():
            for path in [STUDY / f"burnettbench_{iid}_{suffix}.jsonl",
                         DESK  / f"burnettbench_{iid}_{suffix}.jsonl"]:
                if path.exists():
                    for line in open(path):
                        if line.strip():
                            r = json.loads(line)
                            key = (r["model"], r.get("condition","?"), r["fact_id"])
                            if key not in seen:
                                seen.add(key)
                                all_v.append(r)
                    break
        return all_v

    for flag, label in [(False, "OUT_V2 accuracy (goal: >= A)"),
                        (True,  "IN_V2  recall   (goal: ~= V2)")]:
        print(f"\n{label}")
        conds_to_show = [c for c in all_cond_labels if c in verdict_files or any(
            (STUDY / f"burnettbench_{iid}_{c}_verdicts.jsonl").exists() for iid in INSTITUTIONS)]
        print(f"{'Institution':<22}" + "".join(f"{c:>9}" for c in conds_to_show))
        print("─" * (22 + 9 * len(conds_to_show)))

        for iid in INSTITUTIONS:
            all_v = load_verdicts(iid)
            subset = [v for v in all_v if v.get("in_v2") == flag]
            row = f"{INSTITUTIONS[iid]:<22}"
            for cond in conds_to_show:
                recs = [v for v in subset if v.get("condition") == cond]
                row += (f"{sum(1 for v in recs if v['verdict']=='A')/len(recs)*100:>8.1f}%"
                        if recs else f"{'—':>9}")
            print(row)

        print("─" * (22 + 9 * len(conds_to_show)))
        row = f"{'MEAN':<22}"
        for cond in conds_to_show:
            accs = []
            for iid in INSTITUTIONS:
                all_v = load_verdicts(iid)
                subset = [v for v in all_v if v.get("in_v2") == flag]
                recs   = [v for v in subset if v.get("condition") == cond]
                if recs:
                    accs.append(sum(1 for v in recs if v["verdict"]=="A")/len(recs)*100)
            row += f"{sum(accs)/len(accs):>8.1f}%" if accs else f"{'—':>9}"
        print(row)


def main():
    print("Loading SentenceTransformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Load data ──────────────────────────────────────────────────────────────
    synthetic = [json.loads(l)
                 for l in open(STUDY / "burnettbench_opt4_synthetic_all.jsonl")
                 if l.strip()]

    all_data = {}
    gt_texts, gt_labels, gt_iids = [], [], []
    syn_texts, syn_labels, syn_iids = [], [], []

    for iid in INSTITUTIONS:
        v2_text     = (DESK / f"burnettbench_{iid}_llmstxt_v2.txt").read_text()
        annotations = parse_annotations(v2_text)
        probes      = [json.loads(l)
                       for l in open(DESK / f"burnettbench_{iid}_factprobes_full.jsonl")
                       if l.strip()]

        all_data[iid] = {
            "probes":      probes,
            "annotations": annotations,
            "v2_text":     v2_text,
        }

        for p in probes:
            gt_texts.append(make_input(p["probe"], annotations))
            gt_labels.append(int(p.get("in_v2", True)))
            gt_iids.append(iid)

        for s in synthetic:
            if s["institution"] == iid:
                syn_texts.append(make_input(s["question"], annotations))
                syn_labels.append(s["label"])
                syn_iids.append(iid)

    print(f"GT: {len(gt_texts)} examples across {len(INSTITUTIONS)} institutions")
    print(f"Synthetic: {len(syn_texts)} examples")

    # ── Encode ────────────────────────────────────────────────────────────────
    print("\nEncoding GT...")
    gt_embs = embedder.encode(gt_texts, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True)
    print("Encoding synthetic...")
    syn_embs = embedder.encode(syn_texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True)

    # ── CV comparison ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("LEAVE-ONE-OUT CROSS-VALIDATION COMPARISON")

    # OPT5: GT only
    cv_opt5, _ = run_loo_cv(
        "OPT5 — GT only (no synthetic)",
        gt_embs, gt_labels, gt_iids,
        None, None, None,
        denoise_threshold=None
    )

    # OPT4v3 at each threshold
    best_thresh = None
    best_overall = 0.0
    cv_by_thresh = {}

    for thresh in DENOISE_THRESHOLDS:
        cv_res, _ = run_loo_cv(
            f"OPT4v3 — GT + filtered synthetic (conf ≥ {thresh:.2f})",
            gt_embs, gt_labels, gt_iids,
            syn_embs, syn_labels, syn_iids,
            denoise_threshold=thresh
        )
        cv_by_thresh[thresh] = cv_res
        all_true = [l for r in cv_res.values() for l, _ in r]
        all_pred = [p for r in cv_res.values() for _, p in r]
        overall  = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
        if overall > best_overall:
            best_overall = overall
            best_thresh = thresh

    print(f"\nBest OPT4v3 threshold: {best_thresh} (CV acc: {best_overall:.3f})")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if not RUN_BENCHMARK:
        print("\nSkipping benchmark (RUN_BENCHMARK=False)")
        return

    print(f"\nRunning benchmark for OPT5 and OPT4v3 (threshold={best_thresh})...")

    run_benchmark("OPT5", "OPT5",
                  all_data, embedder,
                  gt_embs, gt_labels, gt_iids,
                  None, None, None,
                  denoise_threshold=None)

    best_cond = f"OPT4v3_{int(best_thresh*100)}"
    run_benchmark(f"OPT4v3 (conf≥{best_thresh})", best_cond,
                  all_data, embedder,
                  gt_embs, gt_labels, gt_iids,
                  syn_embs, syn_labels, syn_iids,
                  denoise_threshold=best_thresh)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_summary(["OPT5", best_cond])


if __name__ == "__main__":
    main()
