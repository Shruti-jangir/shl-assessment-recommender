"""
Replays the 10 provided gold conversation traces against a running
instance of our /chat endpoint, and reports:
  - schema compliance per turn
  - whether every returned URL is genuinely from our catalog
  - Recall@10 of our final shortlist vs. the trace's expected final shortlist
  - turn count vs. the 8-turn cap

Usage:
    uvicorn app.main:app --reload &          # in one terminal
    python scripts/eval_traces.py            # in another

Optional: point at a deployed URL instead of localhost:
    python scripts/eval_traces.py --url https://your-app.onrender.com
"""
import argparse
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT / "data" / "sample_conversations" / "GenAI_SampleConversations"
CATALOG_PATH = ROOT / "data" / "catalog_processed.json"


def parse_trace(path: Path):
    text = path.read_text()
    user_turns = re.findall(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?:\n\n|\n\*\*Agent\*\*)", text, re.DOTALL)
    user_turns = [re.sub(r"\s+", " ", u).strip().rstrip(">").strip() for u in user_turns]

    # Grab every markdown table's "Name" column, keep the LAST one as the
    # expected final shortlist (the trace's last agent turn with a table).
    tables = re.findall(r"\|---.*?\n((?:\|.*\n)+)", text)
    expected_names = []
    if tables:
        last_table = tables[-1]
        for row in last_table.strip().split("\n"):
            cols = [c.strip() for c in row.split("|")]
            cols = [c for c in cols if c]
            if len(cols) >= 2 and cols[0].isdigit():
                expected_names.append(cols[1])
    return user_turns, expected_names


def run_trace(client: httpx.Client, base_url: str, user_turns, catalog_urls):
    messages = []
    last_recs = []
    for i, turn in enumerate(user_turns):
        messages.append({"role": "user", "content": turn})
        resp = client.post(f"{base_url}/chat", json={"messages": messages}, timeout=35)
        if resp.status_code != 200:
            return {"error": f"turn {i+1}: HTTP {resp.status_code} {resp.text[:200]}"}
        body = resp.json()
        for field in ("reply", "recommendations", "end_of_conversation"):
            if field not in body:
                return {"error": f"turn {i+1}: missing field '{field}' in response"}
        for r in body["recommendations"]:
            if r["url"] not in catalog_urls:
                return {"error": f"turn {i+1}: hallucinated URL not in catalog: {r['url']}"}
        if not (0 <= len(body["recommendations"]) <= 10):
            return {"error": f"turn {i+1}: recommendations count out of bounds"}
        if body["recommendations"]:
            last_recs = body["recommendations"]
        messages.append({"role": "assistant", "content": body["reply"]})
        if len(messages) >= 8:
            break
    return {"final_recs": last_recs, "turns_used": len(messages)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text())
    catalog_urls = {c["url"] for c in catalog}

    trace_files = sorted(TRACES_DIR.glob("*.md"))
    if not trace_files:
        print(f"No traces found in {TRACES_DIR}")
        sys.exit(1)

    recalls = []
    with httpx.Client() as client:
        try:
            h = client.get(f"{args.url}/health", timeout=10)
            print(f"health check: {h.status_code} {h.json()}")
        except Exception as e:
            print(f"Could not reach {args.url}/health: {e}")
            sys.exit(1)

        for path in trace_files:
            user_turns, expected_names = parse_trace(path)
            if not user_turns:
                print(f"{path.name}: could not parse user turns, skipping")
                continue
            result = run_trace(client, args.url, user_turns, catalog_urls)
            if "error" in result:
                print(f"{path.name}: FAIL - {result['error']}")
                recalls.append(0.0)
                continue
            got_names = {r["name"] for r in result["final_recs"]}
            exp_names = set(expected_names)
            hit = len(got_names & exp_names)
            recall = hit / len(exp_names) if exp_names else None
            if recall is not None:
                recalls.append(recall)
            print(
                f"{path.name}: turns_used={result['turns_used']} "
                f"got={len(got_names)} expected={len(exp_names)} "
                f"overlap={hit} recall={recall}"
            )
            if recall is not None and recall < 1.0:
                print(f"    expected: {sorted(exp_names)}")
                print(f"    got:      {sorted(got_names)}")

    if recalls:
        print(f"\nMean Recall@10 across {len(recalls)} traces: {sum(recalls)/len(recalls):.3f}")


if __name__ == "__main__":
    main()
