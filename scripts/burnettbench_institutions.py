"""
BurnettBench — Institution Config Registry
Each institution defines: id, name, prompts_file, llmstxt_v2, groundtruth_file, adv_prompts_file, keywords
"""
from pathlib import Path

DESK = Path("/Users/eudaimonia/Desktop")

INSTITUTIONS = {
    "bakame": {
        "id": "bakame",
        "name": "Bakame AI",
        "prefix": "BAK",
        "prompts_file":   DESK / "burnettbench_bakame_prompts.jsonl",
        "llmstxt_v2":     DESK / "burnettbench_bakame_llmstxt_v2.txt",
        "groundtruth":    DESK / "burnettbench_bakame_groundtruth.jsonl",
        "adv_prompts":    DESK / "burnettbench_bakame_adv_prompts.jsonl",
        "keywords": [
            "229201520",           # DID number — only in document
            "bakame aviation",     # product name, not just "bakame"
            "bakame moe",          # internal model name
            "niyorurema",          # CEO surname
            "mame diarra",         # CSO name
            "dgx spark",           # specific hardware
            "gb10",                # GPU model
            "128gb",               # specific memory
            "offline-via-telephony",  # exact phrase from doc
            "aviation.bakame.online", # specific URL
            "create grant",        # award name
            "$8,000",              # grant amount
            "lockheed martin ai ethics",  # competition name
            "octofinalist",        # specific result
            "kinyarwanda",         # language (specific enough)
        ],
    },
    "mayo": {
        "id": "mayo",
        "name": "Mayo Clinic",
        "prefix": "MC",
        "prompts_file":   DESK / "burnettbench_mayo_prompts.jsonl",
        "llmstxt_v2":     DESK / "burnettbench_mayo_llmstxt_v2.txt",
        "groundtruth":    DESK / "burnettbench_mayo_groundtruth.jsonl",
        "adv_prompts":    DESK / "burnettbench_mayo_adv_prompts.jsonl",
        "keywords": [
            "farrugia",            # CEO surname — not generic
            "gianrico",            # CEO first name
            "21.5 billion",        # 2025 revenue — specific
            "1.343 billion",       # research investment
            "264.9 million",       # NIH funding
            "161,590",             # surgical cases 2025
            "5.4 million",         # outpatient visits
            "67,900",              # med school tuition
            "jay alix",            # donor name
            "alix school",         # school name after gift
            "$200 million",        # gift amount
            "kirklin",             # surgeon — open-heart pioneer
            "john w. kirklin",     # full name
            "1955",                # open-heart year
            "498 grants",          # NIH grant count
        ],
    },
    "ve": {
        "id": "ve",
        "name": "Vinson & Elkins",
        "prefix": "VE",
        "prompts_file":   DESK / "burnettbench_ve_prompts.jsonl",
        "llmstxt_v2":     DESK / "burnettbench_ve_llmstxt_v2.txt",
        "groundtruth":    DESK / "burnettbench_ve_groundtruth.jsonl",
        "adv_prompts":    DESK / "burnettbench_ve_adv_prompts.jsonl",
        "keywords": [
            "fullenweider",        # chair surname — only in document
            "$1.13 billion",       # 2023 revenue
            "4.51 million",        # PEP
            "$4,327",              # summer associate weekly pay
            "1,900 billable",      # bonus threshold
            "$225,000",            # first-year base salary
            "cravath",             # scale reference
            "stonepeak",           # notable deal client
            "coastal virginia",    # notable deal name
            "2.6 gw",              # deal size
            "talen energy",        # notable deal client
            "1,920 mw",            # deal size
            "$30 million",         # enron settlement
            "keith fullenweider",  # full chair name
            "20,000 pro bono",     # specific pro bono figure
        ],
    },
    "uscis": {
        "id": "uscis",
        "name": "USCIS",
        "prefix": "US",
        "prompts_file":   DESK / "burnettbench_uscis_prompts.jsonl",
        "llmstxt_v2":     DESK / "burnettbench_uscis_llmstxt_v2.txt",
        "groundtruth":    DESK / "burnettbench_uscis_groundtruth.jsonl",
        "adv_prompts":    DESK / "burnettbench_uscis_adv_prompts.jsonl",
        "keywords": [
            "$1,440",              # I-485 paper fee — exact figure
            "$1,390",              # I-485 online fee
            "$760",                # N-400 paper fee
            "$710",                # N-400 online fee
            "$675",                # I-130 paper fee
            "$520",                # I-765 paper fee
            "800-375-5283",        # USCIS phone — only in document
            "my.uscis.gov",        # specific URL
            "13-character",        # receipt number format
            "20,780",              # employee count
            "10.8 million",        # FY2023 completions
            "88 domestic",         # field office count
            "400% federal poverty", # N-400 reduced fee threshold
            "6.1 billion",         # FY2023 budget
            "7.9 months",          # I-140 processing time FY2025
        ],
    },
    "rice": {
        "id": "rice",
        "name": "Rice University",
        "prefix": "RI",
        "prompts_file":   DESK / "burnettbench_rice_prompts.jsonl",
        "llmstxt_v2":     DESK / "burnettbench_rice_llmstxt_v2.txt",
        "groundtruth":    DESK / "burnettbench_rice_groundtruth.jsonl",
        "adv_prompts":    DESK / "burnettbench_rice_adv_prompts.jsonl",
        "keywords": [
            "70,720",              # tuition 2026-27 — exact
            "36,791",              # applications 2025-26
            "2,948",               # admitted students
            "8,880",               # total enrollment
            "4,807",               # undergrad count
            "277.7 million",       # research expenditures FY2024
            "174 million",         # total grant aid FY2026
            "$15,091",             # average net price
            "buckminsterfullerene",# Nobel discovery — specific term
            "chao college",        # 12th residential college name
            "rice investment",     # financial aid program name
            "trustee distinguished scholarship",  # merit award
            "6:1",                 # student-faculty ratio
            "912",                 # full-time faculty count
            "loan-free",           # policy term (since Fall 2022)
        ],
    },
}
