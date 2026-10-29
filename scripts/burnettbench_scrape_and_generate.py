#!/usr/bin/env python3
"""
BurnettBench — OPT4 Data Expansion
Step 1: Scrape institution websites → markdown
Step 2: Generate labeled queries via GPT-4o-mini
  - Label 1 (in scope):  query answerable from llms.txt
  - Label 0 (out scope): query from website but NOT in llms.txt

Output: burnettbench_{iid}_synthetic_queries.jsonl
        burnettbench_opt4_training_data.jsonl  (all institutions combined)

Usage: python burnettbench_scrape_and_generate.py
"""
import json, re, time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import html2text

DESK   = Path("/Users/eudaimonia/Desktop")
STUDY  = Path("/Users/eudaimonia/Desktop/burnettbench_study2")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pages to scrape per institution
SCRAPE_TARGETS = {
    "bakame": {
        "name": "Bakame AI",
        "urls": [
            "https://bakame.online",
            "https://bakame.online/about",
            "https://aviation.bakame.online",
        ]
    },
    "mayo": {
        "name": "Mayo Clinic",
        "urls": [
            "https://en.wikipedia.org/wiki/Mayo_Clinic",
            "https://en.wikipedia.org/wiki/Mayo_Clinic_Alix_School_of_Medicine",
            "https://en.wikipedia.org/wiki/Mayo_Clinic_Health_System",
        ]
    },
    "rice": {
        "name": "Rice University",
        "urls": [
            "https://www.rice.edu/about",
            "https://admission.rice.edu/apply/first-year",
            "https://financialaid.rice.edu/rice-investment",
            "https://www.rice.edu/research",
        ]
    },
    "uscis": {
        "name": "USCIS",
        "urls": [
            "https://www.uscis.gov/about-us",
            "https://www.uscis.gov/forms/all-forms",
            "https://www.uscis.gov/citizenship",
            "https://www.uscis.gov/green-card",
        ]
    },
    "ve": {
        "name": "Vinson & Elkins",
        "urls": [
            "https://www.velaw.com/about/",
            "https://www.velaw.com/careers/",
            "https://en.wikipedia.org/wiki/Vinson_%26_Elkins",
        ]
    },
}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36")
}

# Prompts
QUERY_GEN_PROMPT = """\
You are generating training data for a scope classifier.

Below is a chunk of text scraped from {institution}'s website.
Also shown is the institution's llms.txt document (their AI-readable summary).

Your task: generate {n} factual questions about {institution} based on the scraped text.
For each question, decide if it can be answered from the llms.txt document (label=1) or not (label=0).

Rules:
- Questions must be specific and factual (not vague like "what does X do?")
- label=1: the llms.txt contains enough information to correctly answer the question
- label=0: the question requires information NOT present in the llms.txt
- Aim for roughly half label=1 and half label=0
- Do NOT repeat questions that are already in the existing queries list

Existing queries (do not repeat):
{existing}

Scraped text:
{scraped}

llms.txt:
{llmstxt}

Return a JSON array only, no other text:
[
  {{"question": "...", "label": 0 or 1, "evidence": "brief reason why in/out of scope"}},
  ...
]"""


def scrape_page(url):
    """Fetch URL and convert to clean markdown."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        h = html2text.HTML2Text()
        h.ignore_links     = True
        h.ignore_images    = True
        h.ignore_tables    = False
        h.body_width       = 0
        md = h.handle(resp.text)
        # Clean up excessive blank lines
        md = re.sub(r'\n{3,}', '\n\n', md)
        # Remove nav/cookie boilerplate patterns
        md = re.sub(r'(?i)(cookie|javascript|skip to|back to top).*\n', '', md)
        return md[:6000]  # cap at 6k chars per page
    except Exception as e:
        return f"SCRAPE_ERROR: {e}"


def llm_call(prompt, max_tokens=1500):
    for attempt in range(3):
        try:
            r = requests.post(OR_URL,
                headers={"Authorization": f"Bearer {OR_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "openai/gpt-4o-mini",
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.7, "max_tokens": max_tokens},
                timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2: return f"ERROR: {e}"
            time.sleep(2 ** attempt)


def extract_json(text):
    """Extract JSON array from LLM response."""
    # Find first [ ... ] block
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except Exception:
        return []


def main():
    all_training = []

    for iid, target in SCRAPE_TARGETS.items():
        name    = target["name"]
        urls    = target["urls"]
        llmstxt = (DESK / f"burnettbench_{iid}_llmstxt_v2.txt").read_text()

        # Load existing groundtruth queries to avoid repeats
        existing_queries = []
        gt_file = DESK / f"burnettbench_{iid}_factprobes_full.jsonl"
        if gt_file.exists():
            existing_queries = [json.loads(l)["probe"]
                                for l in open(gt_file) if l.strip()]

        out_file = STUDY / f"burnettbench_{iid}_synthetic_queries.jsonl"
        done_urls = set()
        existing_synthetic = []
        if out_file.exists():
            for line in open(out_file):
                if line.strip():
                    r = json.loads(line)
                    done_urls.add(r.get("source_url", ""))
                    existing_synthetic.append(r)

        print(f"\n{'─'*65}")
        print(f"{name}: {len(urls)} pages | {len(existing_synthetic)} existing synthetic")

        institution_queries = [q["question"] for q in existing_synthetic]

        new_records = []
        for url in urls:
            if url in done_urls:
                print(f"  SKIP (done): {url}")
                continue

            print(f"  Scraping: {url}", end="", flush=True)
            scraped = scrape_page(url)
            if scraped.startswith("SCRAPE_ERROR"):
                print(f" → {scraped}")
                continue
            print(f" → {len(scraped)} chars")

            existing_str = "\n".join(f"- {q}" for q in
                                     (existing_queries + institution_queries)[:30])

            prompt = QUERY_GEN_PROMPT.format(
                institution=name,
                n=20,
                existing=existing_str or "(none yet)",
                scraped=scraped,
                llmstxt=llmstxt[:3000]
            )

            print(f"    Generating queries...", end="", flush=True)
            response = llm_call(prompt)
            items = extract_json(response)

            if not items:
                print(f" → parse failed: {response[:100]}")
                continue

            print(f" → {len(items)} queries")

            for item in items:
                if not isinstance(item, dict) or "question" not in item:
                    continue
                rec = {
                    "institution":  iid,
                    "source_url":   url,
                    "question":     item.get("question", "").strip(),
                    "label":        int(item.get("label", 0)),
                    "evidence":     item.get("evidence", ""),
                    "synthetic":    True
                }
                if rec["question"]:
                    new_records.append(rec)
                    institution_queries.append(rec["question"])

            time.sleep(1)  # be polite to servers

        if new_records:
            with open(out_file, "a") as f:
                for rec in new_records:
                    f.write(json.dumps(rec) + "\n")

        # Load all for this institution
        all_inst = existing_synthetic + new_records
        label_1 = sum(1 for r in all_inst if r["label"] == 1)
        label_0 = sum(1 for r in all_inst if r["label"] == 0)
        print(f"  Total synthetic: {len(all_inst)} "
              f"(in-scope={label_1}, out-scope={label_0})")

        all_training.extend(all_inst)

    # Save combined training file
    combined_file = STUDY / "burnettbench_opt4_synthetic_all.jsonl"
    with open(combined_file, "w") as f:
        for rec in all_training:
            f.write(json.dumps(rec) + "\n")

    total    = len(all_training)
    in_scope = sum(1 for r in all_training if r["label"] == 1)
    out_scope= total - in_scope
    print(f"\n{'='*65}")
    print(f"DONE — {total} synthetic queries saved")
    print(f"  In-scope  (label=1): {in_scope}")
    print(f"  Out-scope (label=0): {out_scope}")
    print(f"  → {combined_file}")
    print(f"\nReview quality before retraining:")
    print(f"  python3 -c \"import json; "
          f"[print(r['label'], r['question']) for r in "
          f"[json.loads(l) for l in open('{combined_file}') if l.strip()]]\"")


if __name__ == "__main__":
    main()
