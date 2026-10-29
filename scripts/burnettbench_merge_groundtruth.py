#!/usr/bin/env python3
"""
BurnettBench — Merge Extended Ground Truth into Main
Combines burnettbench_{iid}_groundtruth.jsonl (original)
with burnettbench_{iid}_groundtruth_extended.jsonl (web-scraped)
into burnettbench_{iid}_groundtruth_full.jsonl

Also prints a summary split:
  - IN_V2 facts: test V2 recall (same as original study)
  - OUT_V2 facts: test out-of-scope accuracy (new study)

Usage: python burnettbench_merge_groundtruth.py [institution_id]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from burnettbench_institutions import INSTITUTIONS

DESK = Path("/Users/eudaimonia/Desktop")


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]

    for iid in targets:
        if iid not in INSTITUTIONS:
            continue
        inst     = INSTITUTIONS[iid]
        orig     = Path(inst["groundtruth"])
        extended = DESK / f"burnettbench_{iid}_groundtruth_extended.jsonl"
        merged   = DESK / f"burnettbench_{iid}_groundtruth_full.jsonl"

        if not orig.exists():
            print(f"SKIP {iid}: missing original groundtruth")
            continue
        if not extended.exists():
            print(f"SKIP {iid}: missing extended file — run expand_groundtruth.py first")
            continue

        orig_facts = [json.loads(l) for l in open(orig) if l.strip()]
        ext_facts  = [json.loads(l) for l in open(extended) if l.strip()]

        # Original facts get in_v2=True by definition (they came from V2 doc)
        for f in orig_facts:
            f.setdefault("in_v2", True)
            f.setdefault("source", "original")

        all_facts = orig_facts + ext_facts

        # Re-index fact_ids sequentially
        prefix = inst["prefix"]
        for i, f in enumerate(all_facts, 1):
            f["fact_id"] = f"{prefix}{i:03d}"

        with open(merged, "w") as out:
            for f in all_facts:
                out.write(json.dumps(f) + "\n")

        in_v2  = [f for f in all_facts if f.get("in_v2")]
        out_v2 = [f for f in all_facts if not f.get("in_v2")]

        print(f"\n{inst['name']}")
        print(f"  Original facts:  {len(orig_facts)}")
        print(f"  Extended facts:  {len(ext_facts)}")
        print(f"  Total merged:    {len(all_facts)}")
        print(f"  ├─ IN_V2:        {len(in_v2)}  (V2 recall study)")
        print(f"  └─ OUT_V2:       {len(out_v2)}  (out-of-scope study)")
        print(f"  Written to:      {merged.name}")


if __name__ == "__main__":
    main()
