#!/usr/bin/env python3
"""
BurnettBench — OPT4_263
DistilBERT fine-tuned scope classifier on 263 clean GT examples (no synthetic).
Condition name: OPT4_263
Run on DGX (CUDA).
"""
import json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import classification_report

HOME  = Path("/home/happy")
DATA  = HOME / "burnettbench_data"
STUDY = HOME / "burnettbench_study2"
STUDY.mkdir(exist_ok=True)

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

CONDITION = "OPT4_263"


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


class ScopeDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, annotations, label = self.examples[idx]
        enc = self.tokenizer(
            query, annotations,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long)
        }


def train_classifier(train_examples, val_examples, epochs=15, batch_size=8, lr=2e-5):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model     = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2)

    train_ds = ScopeDataset(train_examples, tokenizer)
    val_ds   = ScopeDataset(val_examples,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model.to(device)

    best_val_acc = 0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["label"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        model.eval()
        preds, labels_val = [], []
        with torch.no_grad():
            for batch in val_dl:
                out        = model(input_ids=batch["input_ids"].to(device),
                                   attention_mask=batch["attention_mask"].to(device))
                preds      += out.logits.argmax(-1).cpu().tolist()
                labels_val += batch["label"].tolist()

        val_acc = sum(p == l for p, l in zip(preds, labels_val)) / len(labels_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"    Epoch {epoch+1:>2}/{epochs} — loss: {total_loss/len(train_dl):.4f} "
              f"val_acc: {val_acc:.3f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model, tokenizer, device


def predict(model, tokenizer, device, query, annotations, max_length=256):
    enc = tokenizer(query, annotations,
                    max_length=max_length, truncation=True,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device))
    return int(out.logits.argmax(-1).item())


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
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load GT data only ──────────────────────────────────────────────────────
    all_data = {}
    for iid in INSTITUTIONS:
        v2_text     = (DATA / f"burnettbench_{iid}_llmstxt_v2.txt").read_text()
        annotations = parse_annotations(v2_text)
        probes      = [json.loads(l)
                       for l in open(DATA / f"burnettbench_{iid}_factprobes_full.jsonl")
                       if l.strip()]
        gt_examples = [(p["probe"], annotations, int(p.get("in_v2", True)))
                       for p in probes]
        all_data[iid] = {
            "probes":      probes,
            "annotations": annotations,
            "v2_text":     v2_text,
            "gt_examples": gt_examples,
        }
        in_v2  = sum(1 for _, _, l in gt_examples if l == 1)
        out_v2 = sum(1 for _, _, l in gt_examples if l == 0)
        print(f"{iid}: {len(gt_examples)} GT ({in_v2} IN / {out_v2} OUT)")

    total = sum(len(d["gt_examples"]) for d in all_data.values())
    print(f"\nTotal GT pool: {total} examples")

    # ── Leave-one-institution-out CV ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print("LEAVE-ONE-OUT CV — DistilBERT on 263 GT examples")

    cv_results = {}
    for holdout in INSTITUTIONS:
        print(f"\n── Holdout: {INSTITUTIONS[holdout]} ──")
        train_ex = [ex for iid, d in all_data.items() if iid != holdout
                    for ex in d["gt_examples"]]
        val_ex   = all_data[holdout]["gt_examples"]
        print(f"  Train: {len(train_ex)} | Val: {len(val_ex)}")

        model, tokenizer, device = train_classifier(train_ex, val_ex)

        preds  = [predict(model, tokenizer, device, q, a) for q, a, _ in val_ex]
        labels = [l for _, _, l in val_ex]
        acc    = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        print(f"  Best val acc: {acc:.3f}")
        print(classification_report(labels, preds, target_names=["OUT_V2", "IN_V2"]))

        cv_results[holdout] = list(zip(labels, preds))

        # Gate predictions on GT probes
        gt_preds = {}
        for p in all_data[holdout]["probes"]:
            pred = predict(model, tokenizer, device,
                           p["probe"], all_data[holdout]["annotations"])
            gt_preds[p["fact_id"]] = pred
        all_data[holdout]["gate_preds"] = gt_preds

        torch.save({"state_dict": model.state_dict(), "holdout": holdout, "val_acc": acc},
                   STUDY / f"opt4_263_gate_{holdout}.pt")
        del model

    all_true = [l for r in cv_results.values() for l, _ in r]
    all_pred = [p for r in cv_results.values() for _, p in r]
    overall  = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    print(f"\n{'='*65}")
    print(f"Overall CV accuracy: {overall:.3f}")
    print(classification_report(all_true, all_pred, target_names=["OUT_V2", "IN_V2"]))

    print(f"\n{'Institution':<22} {'IN_V2 passed':>14} {'OUT_V2 blocked':>16}")
    for iid in INSTITUTIONS:
        preds  = all_data[iid]["gate_preds"]
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
    print(f"RUNNING BENCHMARK — condition: {CONDITION}")

    for iid in INSTITUTIONS:
        name    = INSTITUTIONS[iid]
        data    = all_data[iid]
        probes  = data["probes"]
        preds   = data["gate_preds"]
        v2_text = data["v2_text"]

        results_file  = STUDY / f"burnettbench_{iid}_{CONDITION}_results.jsonl"
        verdicts_file = STUDY / f"burnettbench_{iid}_{CONDITION}_verdicts.jsonl"

        system_a  = "You are a helpful assistant."
        system_v2 = (f"You are a helpful assistant. Use the following reference "
                     f"information if relevant.\n<reference>\n{v2_text}\n</reference>")

        done_run = load_done(results_file, ["model", "condition", "fact_id"])
        tasks = []
        for mname, mid in MODELS.items():
            for p in probes:
                if (mname, CONDITION, p["fact_id"]) not in done_run:
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
                           "model": mname, "condition": CONDITION,
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
                           "model": r["model"], "condition": CONDITION,
                           "verdict": verdict, "in_v2": r.get("in_v2", True),
                           "gate_inject": r.get("gate_inject", True)}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"    {completed}/{len(judge_tasks)} judged...", flush=True)
        print("  Judge done ✓")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL RESULTS")
    cond_order = ["A", "V2", "OPT2", "OPT4", "OPT4v2", CONDITION]
    suffixes   = {"A": "factprobe_verdicts_full", "V2": "factprobe_verdicts_full",
                  "OPT2": "gating_verdicts", "OPT4": "opt4_verdicts",
                  "OPT4v2": "opt4v2_verdicts", CONDITION: f"{CONDITION}_verdicts"}

    for flag, label in [(False, "OUT_V2 accuracy (goal: >= A=29.5%)"),
                        (True,  "IN_V2 recall   (goal: ~= V2=88.2%)")]:
        print(f"\n{label}")
        print(f"{'Institution':<22}" + "".join(f"{c:>12}" for c in cond_order))
        print("─" * (22 + 12 * len(cond_order)))
        means = {c: [] for c in cond_order}
        for iid in INSTITUTIONS:
            row = f"{INSTITUTIONS[iid]:<22}"
            for c in cond_order:
                suf = suffixes[c]
                fp  = STUDY / f"burnettbench_{iid}_{suf}.jsonl"
                if not fp.exists():
                    row += f"{'—':>12}"; continue
                recs = [json.loads(l) for l in open(fp) if l.strip()]
                recs = [r for r in recs if r.get("condition") == c
                        and r.get("in_v2") == flag]
                if not recs:
                    row += f"{'—':>12}"; continue
                acc = sum(1 for r in recs if r["verdict"] == "A") / len(recs) * 100
                row += f"{acc:>11.1f}%"
                means[c].append(acc)
            print(row)
        print("─" * (22 + 12 * len(cond_order)))
        row = f"{'MEAN':<22}"
        for c in cond_order:
            row += (f"{sum(means[c])/len(means[c]):>11.1f}%"
                    if means[c] else f"{'—':>12}")
        print(row)


if __name__ == "__main__":
    main()
