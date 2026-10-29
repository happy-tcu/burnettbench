#!/usr/bin/env python3
"""
BurnettBench — Failure Mode Analysis
Why do OUT_V2 facts fail? B (confidently wrong) vs C (not addressed)?
Does llms.txt injection change the failure mode?
Which categories are systematically uncovered?
"""
import json, re
from pathlib import Path
from collections import defaultdict

STUDY = Path("/Users/eudaimonia/Desktop/burnettbench_study2")

INSTITUTIONS = {
    "bakame": "Bakame AI",
    "mayo":   "Mayo Clinic",
    "ve":     "Vinson & Elkins",
    "uscis":  "USCIS",
    "rice":   "Rice University",
}

MODELS = ["qwen3-14b", "gpt-4o-mini", "llama-3.1-70b", "claude-haiku-4-5", "gemini-flash"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_verdicts(iid, suffix):
    fp = STUDY / f"burnettbench_{iid}_{suffix}.jsonl"
    if not fp.exists():
        return []
    return [json.loads(l) for l in open(fp) if l.strip()]


def pct(n, d):
    return f"{n/d*100:.1f}%" if d else "—"


def llmstxt_stats(iid):
    fp = STUDY / f"burnettbench_{iid}_llmstxt_v2.txt"
    if not fp.exists():
        return {}
    text = fp.read_text()
    sections = [l for l in text.splitlines() if l.startswith("##")]
    words     = len(text.split())
    chars     = len(text)
    # count usable sections (not "do not disclose")
    usable = [s for s in sections
              if "do not" not in s.lower() and "disclose" not in s.lower()]
    return {"words": words, "chars": chars,
            "sections": len(sections), "usable_sections": len(usable)}


# ── 1. Load all factprobe verdicts (conditions A and V2) ─────────────────────

all_verdicts = []
for iid in INSTITUTIONS:
    vv = load_verdicts(iid, "factprobe_verdicts_full")
    all_verdicts.extend(vv)

print(f"Total verdict records: {len(all_verdicts)}")
print(f"Conditions present:    {sorted(set(r['condition'] for r in all_verdicts))}\n")


# ── 2. Global failure mode breakdown ─────────────────────────────────────────

print("=" * 70)
print("SECTION 1 — GLOBAL FAILURE MODE: OUT_V2 FACTS")
print("=" * 70)
print("Verdict key: A=correct  B=confidently wrong  C=not addressed\n")

for cond in ["A", "V2"]:
    label = "No injection (condition A)" if cond == "A" else "With llms.txt (condition V2)"
    recs  = [r for r in all_verdicts if r["condition"] == cond and not r["in_v2"]]
    n     = len(recs)
    a     = sum(1 for r in recs if r["verdict"] == "A")
    b     = sum(1 for r in recs if r["verdict"] == "B")
    c     = sum(1 for r in recs if r["verdict"] == "C")
    print(f"  {label}  (n={n})")
    print(f"    Correct (A):          {a:4d}  {pct(a,n)}")
    print(f"    Wrong / confident (B):{b:4d}  {pct(b,n)}")
    print(f"    Not addressed (C):    {c:4d}  {pct(c,n)}")
    print()


# ── 3. Per-institution failure modes ─────────────────────────────────────────

print("=" * 70)
print("SECTION 2 — PER-INSTITUTION FAILURE MODES (OUT_V2)")
print("=" * 70)
header = f"{'Institution':<22} {'Cond':>5} {'n':>5} {'A%':>7} {'B%':>7} {'C%':>7}"
print(header)
print("─" * 60)

for iid, name in INSTITUTIONS.items():
    for cond in ["A", "V2"]:
        recs = [r for r in all_verdicts
                if r["institution"] == iid and r["condition"] == cond and not r["in_v2"]]
        n = len(recs)
        if n == 0:
            continue
        a = sum(1 for r in recs if r["verdict"] == "A")
        b = sum(1 for r in recs if r["verdict"] == "B")
        c = sum(1 for r in recs if r["verdict"] == "C")
        print(f"  {name:<20} {cond:>5} {n:>5} {pct(a,n):>7} {pct(b,n):>7} {pct(c,n):>7}")
    print()


# ── 4. Per-model failure modes ────────────────────────────────────────────────

print("=" * 70)
print("SECTION 3 — PER-MODEL FAILURE MODES (OUT_V2, condition V2)")
print("=" * 70)
header = f"{'Model':<22} {'n':>5} {'A%':>7} {'B%':>7} {'C%':>7}"
print(header)
print("─" * 50)

for model in MODELS:
    recs = [r for r in all_verdicts
            if r["model"] == model and r["condition"] == "V2" and not r["in_v2"]]
    n = len(recs)
    if n == 0:
        continue
    a = sum(1 for r in recs if r["verdict"] == "A")
    b = sum(1 for r in recs if r["verdict"] == "B")
    c = sum(1 for r in recs if r["verdict"] == "C")
    print(f"  {model:<20} {n:>5} {pct(a,n):>7} {pct(b,n):>7} {pct(c,n):>7}")
print()


# ── 5. Injection effect: does V2 injection shift B↔C? ────────────────────────

print("=" * 70)
print("SECTION 4 — INJECTION EFFECT ON FAILURE MODE")
print("Does adding llms.txt change HOW models fail on OUT_V2 facts?")
print("=" * 70)
print()

# Aggregate: for each (iid, fact_id, model) pair compare A vs V2 verdict
pairs = defaultdict(dict)
for r in all_verdicts:
    if not r["in_v2"]:
        key = (r["institution"], r["fact_id"], r["model"])
        pairs[key][r["condition"]] = r["verdict"]

# Transition matrix
transitions = defaultdict(int)
for key, conds in pairs.items():
    if "A" in conds and "V2" in conds:
        transitions[(conds["A"], conds["V2"])] += 1

total_pairs = sum(transitions.values())
print(f"  Matched pairs (same fact×model, both conditions): {total_pairs}\n")
print(f"  {'A→V2 transition':<20} {'count':>8} {'%':>7}")
print("  " + "─" * 38)
for (frm, to), cnt in sorted(transitions.items(), key=lambda x: -x[1]):
    label = f"{frm} → {to}"
    print(f"  {label:<20} {cnt:>8} {pct(cnt, total_pairs):>7}")
print()

# Highlight key transitions
stayed_wrong  = transitions.get(("B","B"), 0)
became_wrong  = transitions.get(("C","B"), 0)  # injection made it confidently wrong
became_silent = transitions.get(("B","C"), 0)
fixed         = transitions.get(("B","A"), 0) + transitions.get(("C","A"), 0)

print(f"  Key findings:")
print(f"    Injection made model confidently wrong (C→B): {became_wrong}  {pct(became_wrong, total_pairs)}")
print(f"    Already wrong, stayed wrong         (B→B):   {stayed_wrong}  {pct(stayed_wrong, total_pairs)}")
print(f"    Injection fixed a wrong answer      (B→A):   {transitions.get(('B','A'),0)}  {pct(transitions.get(('B','A'),0), total_pairs)}")
print(f"    Injection fixed a non-answer        (C→A):   {transitions.get(('C','A'),0)}  {pct(transitions.get(('C','A'),0), total_pairs)}")
print()


# ── 6. Category-level coverage analysis ──────────────────────────────────────

print("=" * 70)
print("SECTION 5 — WHICH FACT CATEGORIES ARE SYSTEMATICALLY UNCOVERED?")
print("(OUT_V2 rate per category across all institutions)")
print("=" * 70)

# Load all probes
all_probes = []
for iid in INSTITUTIONS:
    fp = STUDY / f"burnettbench_{iid}_factprobes_full.jsonl"
    if fp.exists():
        probes = [json.loads(l) for l in open(fp) if l.strip()]
        for p in probes:
            p["institution"] = iid
        all_probes.extend(probes)

cat_stats = defaultdict(lambda: {"total": 0, "out_v2": 0})
for p in all_probes:
    cat = p.get("category", "unknown")
    cat_stats[cat]["total"] += 1
    if not p.get("in_v2", True):
        cat_stats[cat]["out_v2"] += 1

print(f"\n  {'Category':<30} {'Total':>7} {'OUT_V2':>8} {'OUT_V2%':>9}")
print("  " + "─" * 58)
for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["out_v2"]/x[1]["total"] if x[1]["total"] else 0):
    t  = s["total"]
    ov = s["out_v2"]
    print(f"  {cat:<30} {t:>7} {ov:>8} {pct(ov,t):>9}")
print()


# ── 7. Category × failure mode (V2 condition, OUT_V2 facts) ──────────────────

print("=" * 70)
print("SECTION 6 — CATEGORY × FAILURE MODE (V2 condition, OUT_V2 facts)")
print("=" * 70)
print(f"\n  {'Category':<30} {'n':>5} {'A%':>7} {'B%':>7} {'C%':>7}")
print("  " + "─" * 58)

cat_verdicts = defaultdict(list)
for r in all_verdicts:
    if r["condition"] == "V2" and not r["in_v2"]:
        cat_verdicts[r.get("category", "unknown")].append(r["verdict"])

for cat, vv in sorted(cat_verdicts.items(), key=lambda x: -len(x[1])):
    n = len(vv)
    a = vv.count("A")
    b = vv.count("B")
    c = vv.count("C")
    print(f"  {cat:<30} {n:>5} {pct(a,n):>7} {pct(b,n):>7} {pct(c,n):>7}")
print()


# ── 8. llms.txt format vs coverage ───────────────────────────────────────────

print("=" * 70)
print("SECTION 7 — llms.txt FORMAT vs COVERAGE")
print("=" * 70)
print(f"\n  {'Institution':<22} {'Words':>7} {'Sections':>10} {'IN_V2%':>8} {'OUT_V2%':>9}")
print("  " + "─" * 62)

for iid, name in INSTITUTIONS.items():
    stats = llmstxt_stats(iid)
    probes = [p for p in all_probes if p["institution"] == iid]
    n      = len(probes)
    in_v2  = sum(1 for p in probes if p.get("in_v2", True))
    out_v2 = n - in_v2
    print(f"  {name:<22} {stats.get('words',0):>7} {stats.get('usable_sections',0):>10} "
          f"{pct(in_v2,n):>8} {pct(out_v2,n):>9}")
print()

# Correlation note
print("  Correlation (words vs IN_V2 coverage rate):")
pairs_stat = []
for iid, name in INSTITUTIONS.items():
    stats  = llmstxt_stats(iid)
    probes = [p for p in all_probes if p["institution"] == iid]
    n      = len(probes)
    in_v2  = sum(1 for p in probes if p.get("in_v2", True))
    if n:
        pairs_stat.append((stats.get("words", 0), in_v2 / n, name))

if len(pairs_stat) >= 2:
    import statistics
    xs = [p[0] for p in pairs_stat]
    ys = [p[1] for p in pairs_stat]
    xm = statistics.mean(xs); ym = statistics.mean(ys)
    num = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
    den = (sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys)) ** 0.5
    r   = num/den if den else 0
    print(f"    r = {r:.3f}  (positive = longer llms.txt → better coverage)")
    for words, cov, name in sorted(pairs_stat, key=lambda x: x[0]):
        print(f"      {name:<22} {words:>6} words  {cov*100:.1f}% covered")
print()


# ── 9. IN_V2 fact failure modes (sanity check) ───────────────────────────────

print("=" * 70)
print("SECTION 8 — IN_V2 FACT FAILURE MODES (sanity check)")
print("Should be mostly A with V2, worse without")
print("=" * 70)
for cond in ["A", "V2"]:
    recs = [r for r in all_verdicts if r["condition"] == cond and r["in_v2"]]
    n    = len(recs)
    a    = sum(1 for r in recs if r["verdict"] == "A")
    b    = sum(1 for r in recs if r["verdict"] == "B")
    c    = sum(1 for r in recs if r["verdict"] == "C")
    print(f"  Condition {cond} (n={n}):  A={pct(a,n)}  B={pct(b,n)}  C={pct(c,n)}")
print()

print("Analysis complete.")
