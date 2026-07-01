"""
Normalizes data/shl_product_catalog.json (raw scrape) into
data/catalog_processed.json — a clean list the rest of the app consumes.

Run once (or whenever the raw catalog changes):
    python scripts/preprocess_catalog.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "shl_product_catalog.json"
OUT_PATH = ROOT / "data" / "catalog_processed.json"

# SHL's standard test-type letter codes, keyed by the category label
# used in the raw catalog's "keys" field.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


# One confirmed scraping defect in the raw source data: this item's name lost
# the word "Excel" between "Microsoft" and "365" (likely an icon that didn't
# render as text on SHL's page). Confirmed via URL slug cross-check against
# the rest of the catalog — this is the only entry affected. Fixed by id
# rather than by pattern-matching text, so it doesn't risk mangling anything
# else if the raw catalog changes.
NAME_FIXES = {
    "4207": "Microsoft Excel 365 (New)",
}


def parse_duration_minutes(duration: str):
    if not duration:
        return None
    m = re.search(r"(\d+)", duration)
    return int(m.group(1)) if m else None


def build_embedding_text(entry: dict) -> str:
    """Text that gets embedded for semantic retrieval. Front-load the name
    and job levels/skills-bearing signal since those matter most for match."""
    parts = [
        entry["name"],
        entry.get("description", ""),
        "Categories: " + ", ".join(entry.get("keys", [])),
        "Job levels: " + ", ".join(entry.get("job_levels", [])),
    ]
    return "\n".join(p for p in parts if p)


def main():
    raw = json.loads(RAW_PATH.read_text(), strict=False)

    processed = []
    seen_ids = set()
    for e in raw:
        if e.get("status") != "ok":
            continue
        eid = e["entity_id"]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)

        codes = sorted({CATEGORY_TO_CODE.get(k) for k in e.get("keys", []) if CATEGORY_TO_CODE.get(k)})
        # Collapse literal newlines that show up