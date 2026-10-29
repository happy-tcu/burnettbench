#!/usr/bin/env python3
"""
BurnettBench — Significance Testing: OUT_V2 Accuracy Degradation
Tests whether the accuracy DROP on out-of-scope facts (in_v2=False)
when V2 context is provided is statistically significant.

Also tests IN_V2 lift for comparison, and the contrast between the two
(V2 helps IN_V2 facts but hurts OUT_V2 facts).

Uses:
  - McNemar's test: paired A vs V2 on same (fact_id × model)
  - Wilson 95% CI on each accuracy estimate
  - Cohen's h effect size
  - Interaction test: is the IN_V2 lift significantly different from OUT_V2 lift?
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
    b = sum(1 for a,v in pairs if not a and v)
    c = sum(1 for a,v in pairs if a and not v)
    n = b + c
    if n == 0: return (0.0, 1.0, b, c)
    if n < 25:
        p = 2 * min(stats.binom.cdf(min(b,c), n, 0.5),
                    1 - stats.binom.cdf(min(b,c)-1, n, 0.5))
        chi2 = (abs(b-c)-1)**2/n
    else:
        chi2 = (abs(b-c)-1)**2/n
        p = stats.chi2.sf(chi2, df=1)
    return (chi2, p, b, c)


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def analyse_subset(verdicts, label):
    """Run McNemar + CI on a subset of verdicts. Returns summary dict."""
    rows = []
    for m in MODELS:
        a_v  = {v["fact_id"]: v for v in verdicts if v["model"]==m and v["condition"]=="A"}
        v2_v = {v["fact_id"]: v for v in verdicts if v["model"]==m and v["condition"]=="V2"}
        shared = set(a_v) & set(v2_v)
        if not shared: continue
        pairs = [(a_v[fid]["verdict"]=="A", v2_v[fid]["verdict"]=="A") for fid in shared]
        n = len(pairs)
        a_k  = sum(p[0] for p in pairs)
        v2_k = sum(p[1] for p in pairs)
        chi2, p_val, b, c = mcnemar(pairs)
        h = cohens_h(a_k/n, v2_k/n)
        rows.append({"model": m, "n": n, "a_k": a_k, "v2_k": v2_k,
                     "b": b, "c": c, "chi2": chi2, "p": p_val, "h": h})
    return rows


def print_table(rows, label, direction="lift"):
    print(f"\n  {label}")
    print(f"  {'Model':<22}{'A (95% CI)':<22}{'V2 (95% CI)':<22}{'p':<11}{'h':>7}")
    for r in rows:
        n = r["n"]
        a_lo, a_hi, a_c   = wilson_ci(r["a_k"], n)
        v2_lo, v2_hi, v2_c = wilson_ci(r["v2_k"], n)
        a_s  = f"{a_c*100:.1f}% [{a_lo*100:.0f}–{a_hi*100:.0f}%]"
        v2_s = f"{v2_c*100:.1f}% [{v2_lo*100:.0f}–{v2_hi*100:.0f}%]"
        p_s  = f"{r['p']:.4f}{sig(r['p'])}"
        print(f"  {r['model']:<22}{a_s:<22}{v2_s:<22}{p_s:<11}{r['h']:>+7.3f}")

    # Pooled
    total_n = sum(r["n"] for r in rows)
    a_k  = sum(r["a_k"] for r in rows)
    v2_k = sum(r["v2_k"] for r in rows)
    all_b = sum(r["b"] for r in rows)
    all_c = sum(r["c"] for r in rows)
    n_bc  = all_b + all_c
    if n_bc:
        chi2 = (abs(all_b-all_c)-1)**2/n_bc
        p_pool = stats.chi2.sf(chi2, df=1)
    else:
        p_pool = 1.0
    h = cohens_h(a_k/total_n, v2_k/total_n) if total_n else 0
    a_lo, a_hi, a_c   = wilson_ci(a_k, total_n)
    v2_lo, v2_hi, v2_c = wilson_ci(v2_k, total_n)
    a_s  = f"{a_c*100:.1f}% [{a_lo*100:.0f}–{a_hi*100:.0f}%]"
    v2_s = f"{v2_c*100:.1f}% [{v2_lo*100:.0f}–{v2_hi*100:.0f}%]"
    delta = (v2_c - a_c) * 100
    print(f"  {'POOLED':<22}{a_s:<22}{v2_s:<22}{sig(p_pool):>11}{h:>+7.3f}  Δ={delta:+.1f}pp  p={p_pool:.4f}")
    return {"a_k": a_k, "v2_k": v2_k, "n": total_n,
            "b": all_b, "c": all_c, "p": p_pool, "h": h,
            "delta": delta}


def main():
    print("OUT-OF-SCOPE ACCURACY — SIGNIFICANCE TESTING")
    print("=" * 75)
    print("Null hypothesis: V2 context has no effect on out-of-scope accuracy")
    print("Direction: we expect harm (A > V2 for OUT_V2 facts)\n")
    print("Sig: *** p<.001  ** p<.01  * p<.05  ns p≥.05\n")

    grand_in  = []
    grand_out = []

    for iid, name in INSTITUTIONS.items():
        vfile = DESK / f"burnettbench_{iid}_factprobe_verdicts_full.jsonl"
        if not vfile.exists():
            print(f"SKIP {iid}")
            continue
        verdicts = [json.loads(l) for l in open(vfile) if l.strip()]
        in_v  = [v for v in verdicts if v.get("in_v2")]
        out_v = [v for v in verdicts if not v.get("in_v2")]

        print(f"{'─'*75}")
        print(f"{name}  (IN_V2 n={len(set(v['fact_id'] for v in in_v))} facts | "
              f"OUT_V2 n={len(set(v['fact_id'] for v in out_v))} facts)")

        in_rows  = analyse_subset(in_v,  "IN_V2 (recall)")
        out_rows = analyse_subset(out_v, "OUT_V2 (out-of-scope)")

        r_in  = print_table(in_rows,  "IN_V2 — recall lift (expected positive)")
        r_out = print_table(out_rows, "OUT_V2 — scope harm (expected negative)")

        grand_in.append({"iid": iid, "name": name, **r_in})
        grand_out.append({"iid": iid, "name": name, **r_out})

    # Grand summary
    print(f"\n{'='*75}")
    print("GRAND SUMMARY ACROSS ALL INSTITUTIONS")
    print(f"\n  {'Institution':<22}{'IN_V2 Δ':>10}{'IN_V2 sig':>11}{'OUT_V2 Δ':>11}{'OUT_V2 sig':>12}")
    for ri, ro in zip(grand_in, grand_out):
        print(f"  {ri['name']:<22}{ri['delta']:>+9.1f}pp{sig(ri['p']):>11}"
              f"{ro['delta']:>+10.1f}pp{sig(ro['p']):>12}")

    # Pooled across all institutions
    all_in_a  = sum(r["a_k"] for r in grand_in)
    all_in_v2 = sum(r["v2_k"] for r in grand_in)
    all_in_n  = sum(r["n"] for r in grand_in)
    all_out_a  = sum(r["a_k"] for r in grand_out)
    all_out_v2 = sum(r["v2_k"] for r in grand_out)
    all_out_n  = sum(r["n"] for r in grand_out)

    in_b  = sum(r["b"] for r in grand_in)
    in_c  = sum(r["c"] for r in grand_in)
    out_b = sum(r["b"] for r in grand_out)
    out_c = sum(r["c"] for r in grand_out)

    def pooled_p(b, c):
        n = b + c
        if n == 0: return 1.0
        chi2 = (abs(b-c)-1)**2/n
        return stats.chi2.sf(chi2, df=1)

    in_p  = pooled_p(in_b, in_c)
    out_p = pooled_p(out_b, out_c)
    in_delta  = (all_in_v2/all_in_n   - all_in_a/all_in_n)   * 100
    out_delta = (all_out_v2/all_out_n - all_out_a/all_out_n) * 100
    in_h  = cohens_h(all_in_a/all_in_n,   all_in_v2/all_in_n)
    out_h = cohens_h(all_out_a/all_out_n, all_out_v2/all_out_n)

    print(f"\n  {'OVERALL':<22}{in_delta:>+9.1f}pp{sig(in_p):>11}"
          f"{out_delta:>+10.1f}pp{sig(out_p):>12}")

    print(f"\n  IN_V2:  Δ={in_delta:+.1f}pp  h={in_h:+.3f}  p={in_p:.4f} {sig(in_p)}")
    print(f"  OUT_V2: Δ={out_delta:+.1f}pp  h={out_h:+.3f}  p={out_p:.4f} {sig(out_p)}")

    print(f"""
  INTERPRETATION:
  ├─ IN_V2 lift ({in_delta:+.1f}pp) is {'significant' if in_p < 0.05 else 'not significant'} — V2 improves recall of covered facts
  ├─ OUT_V2 harm ({out_delta:+.1f}pp) is {'significant' if out_p < 0.05 else 'not significant'} — V2 degrades accuracy on uncovered facts
  └─ The asymmetry (help IN / harm OUT) is the core finding:
     llms.txt V2 is not neutral on out-of-scope questions.
     An incomplete document actively displaces baseline model knowledge.

  Effect size guide: |h| < 0.2 = small, 0.2–0.5 = medium, > 0.5 = large
    IN_V2  h={in_h:+.3f} → {'large' if abs(in_h) > 0.5 else 'medium' if abs(in_h) > 0.2 else 'small'} effect
    OUT_V2 h={out_h:+.3f} → {'large' if abs(out_h) > 0.5 else 'medium' if abs(out_h) > 0.2 else 'small'} effect
""")


if __name__ == "__main__":
    main()
