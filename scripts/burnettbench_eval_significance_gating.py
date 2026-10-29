#!/usr/bin/env python3
"""
BurnettBench — Significance Testing: Gating Conditions
Tests whether OPT2 and OPT3 significantly recover OUT_V2 accuracy
and whether they significantly preserve IN_V2 recall vs V2 baseline.

Comparisons:
  OPT2 vs A   — does OPT2 match baseline?
  OPT2 vs V2  — does OPT2 recover the harm?
  OPT3 vs A
  OPT3 vs V2
"""
import json, math
from pathlib import Path
from scipy import stats

DESK = Path("/Users/eudaimonia/Desktop")
INSTITUTIONS = {
    "bakame": "Bakame AI",
    "mayo":   "Mayo Clinic",
    "ve":     "Vinson & Elkins",
    "uscis":  "USCIS",
    "rice":   "Rice University",
}
MODELS = ["qwen3-14b","gpt-4o-mini","llama-3.1-70b","claude-haiku-4-5","gemini-flash"]


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (max(0, center-margin), min(1, center+margin), center)


def cohens_h(p1, p2):
    p1 = max(1e-9, min(1-1e-9, p1))
    p2 = max(1e-9, min(1-1e-9, p2))
    return 2*math.asin(math.sqrt(p2)) - 2*math.asin(math.sqrt(p1))


def mcnemar(pairs):
    b = sum(1 for a, v in pairs if not a and v)
    c = sum(1 for a, v in pairs if a and not v)
    n = b + c
    if n == 0: return (0.0, 1.0, b, c)
    if n < 25:
        p = 2 * min(stats.binom.cdf(min(b,c), n, 0.5),
                    1 - stats.binom.cdf(min(b,c)-1, n, 0.5))
        chi2 = (abs(b-c)-1)**2/n if n > 0 else 0
    else:
        chi2 = (abs(b-c)-1)**2/n
        p = stats.chi2.sf(chi2, df=1)
    return (chi2, p, b, c)


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def compare_conditions(verdicts, cond_a, cond_b):
    """McNemar pooled across all models. Returns summary dict."""
    all_b, all_c, tot_ak, tot_bk, tot_n = 0, 0, 0, 0, 0
    for m in MODELS:
        va = {v["fact_id"]: v for v in verdicts if v["model"]==m and v["condition"]==cond_a}
        vb = {v["fact_id"]: v for v in verdicts if v["model"]==m and v["condition"]==cond_b}
        shared = set(va) & set(vb)
        if not shared: continue
        pairs = [(va[fid]["verdict"]=="A", vb[fid]["verdict"]=="A") for fid in shared]
        n = len(pairs)
        tot_n  += n
        tot_ak += sum(p[0] for p in pairs)
        tot_bk += sum(p[1] for p in pairs)
        b = sum(1 for a,v in pairs if not a and v)
        c = sum(1 for a,v in pairs if a and not v)
        all_b += b; all_c += c
    if tot_n == 0: return None
    n_bc = all_b + all_c
    p_pool = stats.chi2.sf((abs(all_b-all_c)-1)**2/n_bc, df=1) if n_bc else 1.0
    h = cohens_h(tot_ak/tot_n, tot_bk/tot_n)
    delta = (tot_bk/tot_n - tot_ak/tot_n)*100
    return {"delta": delta, "p": p_pool, "h": h,
            "b": all_b, "c": all_c, "n": tot_n,
            "a_k": tot_ak, "b_k": tot_bk}


def main():
    print("GATING SIGNIFICANCE TESTING")
    print("=" * 72)
    print("Sig: *** p<.001  ** p<.01  * p<.05  ns p≥.05\n")

    grand = {"OPT2": {"in": [], "out": []}, "OPT3": {"in": [], "out": []}}

    for iid, name in INSTITUTIONS.items():
        orig_file = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
        gate_file = DESK / f"burnettbench_{iid}_gating_verdicts.jsonl"
        if not orig_file.exists() or not gate_file.exists():
            print(f"SKIP {iid}")
            continue

        orig  = [json.loads(l) for l in open(orig_file) if l.strip()]
        gate  = [json.loads(l) for l in open(gate_file) if l.strip()]
        all_v = orig + gate

        in_v  = [v for v in all_v if v.get("in_v2")]
        out_v = [v for v in all_v if not v.get("in_v2")]

        print(f"{'─'*72}")
        print(f"{name}  (IN_V2 n={len(set(v['fact_id'] for v in in_v))} | "
              f"OUT_V2 n={len(set(v['fact_id'] for v in out_v))})")

        for subset_label, subset, key in [
            ("IN_V2  (recall — OPT should ≈ V2)", in_v, "in"),
            ("OUT_V2 (scope harm — OPT should ≈ A)", out_v, "out"),
        ]:
            print(f"\n  {subset_label}")
            print(f"  {'Comparison':<22}{'Δ':>9}  {'p':>12}  {'h':>8}  note")
            for opt in ["OPT2", "OPT3"]:
                for base, base_lbl in [("A","vs A"), ("V2","vs V2")]:
                    r = compare_conditions(subset, base, opt)
                    if r is None: continue
                    note = ""
                    if key == "out" and base == "A":
                        note = "recovered ✓" if abs(r["delta"]) < 3 and r["p"] > 0.05 else ("partial" if r["delta"] > -5 else "still hurt")
                    elif key == "in" and base == "V2":
                        note = "preserved ✓" if r["delta"] > -5 and r["p"] > 0.05 else "degraded ✗"
                    print(f"  {opt} {base_lbl:<16}{r['delta']:>+8.1f}pp  "
                          f"{r['p']:.4f}{sig(r['p']):>4}  {r['h']:>+7.3f}  {note}")
                r_a  = compare_conditions(subset, "A",  opt)
                r_v2 = compare_conditions(subset, "V2", opt)
                if r_a and r_v2:
                    grand[opt][key].append({"iid": iid, "name": name,
                                            "vs_a": r_a, "vs_v2": r_v2})

    # Grand summary
    print(f"\n{'='*72}")
    print("GRAND SUMMARY — pooled across all 5 institutions\n")
    print(f"  {'Subset':<10}{'Opt':<8}{'Base':<8}{'Δ':>9}  {'p':>12}  {'h':>8}  verdict")
    for opt in ["OPT2","OPT3"]:
        for key, label in [("out","OUT_V2"),("in","IN_V2")]:
            entries = grand[opt][key]
            if not entries: continue
            for base_key, base_lbl in [("vs_a","vs A"),("vs_v2","vs V2")]:
                all_b  = sum(e[base_key]["b"]   for e in entries)
                all_c  = sum(e[base_key]["c"]   for e in entries)
                tot_n  = sum(e[base_key]["n"]   for e in entries)
                tot_ak = sum(e[base_key]["a_k"] for e in entries)
                tot_bk = sum(e[base_key]["b_k"] for e in entries)
                n_bc = all_b + all_c
                p = stats.chi2.sf((abs(all_b-all_c)-1)**2/n_bc, df=1) if n_bc else 1.0
                delta = (tot_bk/tot_n - tot_ak/tot_n)*100 if tot_n else 0
                h = cohens_h(tot_ak/tot_n, tot_bk/tot_n) if tot_n else 0
                verdict = ""
                if label=="OUT_V2" and base_lbl=="vs A":
                    verdict = "recovered ✓" if abs(delta)<3 and p>0.05 else "partial" if delta>-5 else "still hurt"
                elif label=="IN_V2" and base_lbl=="vs V2":
                    verdict = "preserved ✓" if delta>-5 and p>0.05 else "degraded ✗"
                print(f"  {label:<10}{opt:<8}{base_lbl:<8}{delta:>+8.1f}pp  "
                      f"{p:.4f}{sig(p):>4}  {h:>+7.3f}  {verdict}")

    print(f"""
INTERPRETATION:
  OPT2 OUT_V2 vs A  → ns + small Δ = gating fully recovers baseline
  OPT2 OUT_V2 vs V2 → * or better  = gating significantly beats V2 harm
  OPT2 IN_V2  vs V2 → ns           = recall preserved
  OPT3 IN_V2  vs V2 → * or worse   = recall degraded (section truncation)
""")


if __name__ == "__main__":
    main()
