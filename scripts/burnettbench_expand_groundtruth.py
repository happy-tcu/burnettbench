#!/usr/bin/env python3
"""
BurnettBench — Ground Truth Expansion via Web Scraping
Fetches official pages for each institution, extracts atomic verifiable facts
via GPT-4o-mini, deduplicates against existing ground truth, and classifies
each new fact as IN_V2 (covered by the V2 doc) or OUT_OF_V2 (not covered).

OUT_OF_V2 facts enable a new study: does providing V2 context cause models
to hallucinate on questions the V2 document doesn't address?

Usage: python burnettbench_expand_groundtruth.py [institution_id]
       (omit institution_id to run all)
"""
import sys, json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from burnettbench_institutions import INSTITUTIONS

OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
DESK   = Path("/Users/eudaimonia/Desktop")

# All sources per institution — Wikipedia HTML + official pages
SOURCES = {
    "bakame": [
        "https://bakame.online",
        "https://aviation.bakame.online",
    ],
    "mayo": [
        "https://en.wikipedia.org/wiki/Mayo_Clinic",
        "https://en.wikipedia.org/wiki/Mayo_Clinic_Alix_School_of_Medicine",
        "https://en.wikipedia.org/wiki/Mayo_Clinic_Health_System",
    ],
    "ve": [
        "https://en.wikipedia.org/wiki/Vinson_%26_Elkins",
        "https://www.velaw.com/",
        "https://www.velaw.com/careers/",
    ],
    "uscis": [
        "https://en.wikipedia.org/wiki/United_States_Citizenship_and_Immigration_Services",
        "https://www.uscis.gov/about-us",
        "https://www.uscis.gov/forms/filing-fees",
        "https://www.uscis.gov/citizenship",
    ],
    "rice": [
        "https://en.wikipedia.org/wiki/Rice_University",
        "https://www.rice.edu/about",
        "https://financialaid.rice.edu/rice-investment",
        "https://research.rice.edu/",
    ],
}

EXTRACT_PROMPT = """\
You are extracting verifiable atomic facts from a web page about {institution}.

Rules:
- Each fact must be a single, specific, independently verifiable claim
- PREFER facts containing: numbers, dates, dollar amounts, names, rankings, specific quantities
- AVOID: vague claims, opinions, marketing language, anything that cannot be verified
- Each fact must be ONE sentence, under 25 words
- Do NOT include facts already in this list:
{existing_facts}

Web page content:
<content>
{content}
</content>

Return a JSON array of objects. Each object must have:
  "fact": the atomic fact as a single sentence
  "category": one short word label (e.g. financials, history, enrollment, fees, research, leadership, etc.)

Return ONLY valid JSON. No explanation, no markdown fences."""

COVERAGE_PROMPT = """\
Does the following reference document explicitly state, mention, or clearly imply this fact?

Reference document:
<doc>
{v2_text}
</doc>

Fact: "{fact}"

Reply with exactly one word: YES or NO"""


def fetch_page(url, timeout=20):
    """Fetch URL, return clean text (BeautifulSoup stripped)."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0 (research bot)"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "iframe", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text[:12000]
    except Exception as e:
        print(f"    FETCH ERROR {url}: {e}")
        return ""




def gpt_call(prompt, max_tokens=1500):
    for attempt in range(3):
        try:
            r = requests.post(OR_URL,
                headers={"Authorization": f"Bearer {OR_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "openai/gpt-4o-mini",
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0, "max_tokens": max_tokens},
                timeout=45)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2: return None
            time.sleep(2 ** attempt)


def extract_facts_from_page(content, institution_name, existing_facts_text):
    prompt = EXTRACT_PROMPT.format(
        institution=institution_name,
        existing_facts=existing_facts_text,
        content=content,
    )
    raw = gpt_call(prompt)
    if not raw:
        return []
    # Strip markdown fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)
                    and "fact" in d and "category" in d]
    except Exception:
        pass
    return []


def is_covered_by_v2(fact, v2_text):
    prompt = COVERAGE_PROMPT.format(v2_text=v2_text, fact=fact)
    result = gpt_call(prompt, max_tokens=5)
    if result:
        return result.strip().upper().startswith("Y")
    return False


def deduplicate(new_facts, existing_facts_set):
    """Remove facts semantically similar to existing ones using simple overlap."""
    unique = []
    for f in new_facts:
        fact_lower = f["fact"].lower()
        # Skip if any existing fact shares 5+ consecutive words
        words = fact_lower.split()
        is_dup = False
        for ex in existing_facts_set:
            ex_lower = ex.lower()
            for i in range(len(words) - 4):
                phrase = " ".join(words[i:i+5])
                if phrase in ex_lower:
                    is_dup = True
                    break
            if is_dup:
                break
        if not is_dup:
            unique.append(f)
    return unique


def main():
    targets = list(INSTITUTIONS.keys()) if len(sys.argv) < 2 else [sys.argv[1]]
    targets = [t for t in targets if t in INSTITUTIONS]

    for iid in targets:
        inst    = INSTITUTIONS[iid]
        name    = inst["name"]
        v2_text = open(inst["llmstxt_v2"]).read()
        gt_file = Path(inst["groundtruth"])
        out_file = DESK / f"burnettbench_{iid}_groundtruth_extended.jsonl"

        existing = [json.loads(l) for l in open(gt_file) if l.strip()]
        existing_facts_text = "\n".join(f"- {e['fact']}" for e in existing)
        existing_set = {e["fact"] for e in existing}

        # Load already-processed facts to allow resuming
        done_facts = set()
        if out_file.exists():
            for l in open(out_file):
                if l.strip():
                    done_facts.add(json.loads(l)["fact"])

        sources = SOURCES.get(iid, [])
        print(f"\n{'='*65}")
        print(f"{name} — {len(sources)} sources | {len(existing)} existing facts")

        all_new = []

        for source_id in sources:
            print(f"\n  Fetching: {source_id}")
            content = fetch_page(source_id)

            if not content:
                print(f"    Empty — skip")
                continue
            print(f"    {len(content)} chars — extracting facts...")

            combined_existing = existing_facts_text
            if all_new:
                combined_existing += "\n" + "\n".join(f"- {f['fact']}" for f in all_new)

            facts = extract_facts_from_page(content, name, combined_existing)
            facts = deduplicate(facts, existing_set | {f["fact"] for f in all_new})
            print(f"    {len(facts)} new candidate facts")
            all_new.extend(facts)

        if not all_new:
            print(f"  No new facts found for {name}")
            continue

        # Filter out already processed
        to_process = [f for f in all_new if f["fact"] not in done_facts]
        print(f"\n  Checking V2 coverage for {len(to_process)} facts...")

        # Assign IDs starting after existing
        prefix = inst["prefix"]
        next_id = len(existing) + len(done_facts) + 1

        completed = 0
        with open(out_file, "a") as f:
            with ThreadPoolExecutor(max_workers=10) as ex:
                fmap = {
                    ex.submit(is_covered_by_v2, fact_obj["fact"], v2_text): fact_obj
                    for fact_obj in to_process
                }
                for fut in as_completed(fmap):
                    fact_obj = fmap[fut]
                    covered  = fut.result()
                    rec = {
                        "fact_id":  f"{prefix}{next_id:03d}",
                        "category": fact_obj["category"],
                        "fact":     fact_obj["fact"],
                        "in_v2":    covered,
                        "source":   "web",
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    next_id  += 1
                    completed += 1

        # Summary
        all_extended = [json.loads(l) for l in open(out_file) if l.strip()]
        in_v2  = [r for r in all_extended if r["in_v2"]]
        out_v2 = [r for r in all_extended if not r["in_v2"]]

        print(f"\n  Results for {name}:")
        print(f"    Existing GT facts:     {len(existing)}")
        print(f"    New facts extracted:   {len(all_extended)}")
        print(f"    ├─ IN_V2  (V2 covers): {len(in_v2)}")
        print(f"    └─ OUT_V2 (V2 misses): {len(out_v2)}")
        print(f"    Total usable GT:       {len(existing) + len(all_extended)}")

        # Category breakdown
        cats = {}
        for r in all_extended:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(f"\n  New fact categories: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")

    print(f"\n{'='*65}")
    print("DONE — review extended files before merging into main ground truth")
    print("Next step: run burnettbench_merge_groundtruth.py to combine")


if __name__ == "__main__":
    main()
