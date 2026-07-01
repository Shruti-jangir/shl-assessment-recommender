"""
Computes an embedding for every catalog item and caches it to
data/catalog_embeddings.npy (+ data/catalog_embeddings_ids.json to keep
the row order mapped to catalog item ids).

This is a ONE-TIME (or "whenever the catalog changes") offline step —
the deployed service just loads the cached vectors, so a cold start on
Render doesn't need to hit the embeddings API for the whole catalog.

Resumable: if it gets interrupted (e.g. a rate-limit error), just run it
again — it skips items that were already embedded in a prior run.

Run locally:
    export GEMINI_API_KEY=your_key_here
    python scripts/embed_catalog.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog_processed.json"
EMB_PATH = ROOT / "data" / "catalog_embeddings.npy"
IDS_PATH = ROOT / "data" / "catalog_embeddings_ids.json"
CHECKPOINT_PATH = ROOT / "data" / "_embed_checkpoint.json"  # temp, deleted on success

EMBED_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 10  # free tier allows ~100 embedding items/minute; keep batches small
SLEEP_BETWEEN_BATCHES = 8  # seconds; 10 items every 8s stays under 100/min with margin
MAX_RETRIES_PER_BATCH = 6


def embed_batch_with_retry(batch):
    delay = 10
    for attempt in range(1, MAX_RETRIES_PER_BATCH + 1):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except ResourceExhausted:
            print(f"  Rate limited, waiting {delay}s before retry {attempt}/{MAX_RETRIES_PER_BATCH}...")
            time.sleep(delay)
            delay = min(delay * 1.7, 60)
    raise RuntimeError("Exceeded max retries for a batch due to persistent rate limiting.")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running this script.")
    genai.configure(api_key=api_key)

    catalog = json.loads(CATALOG_PATH.read_text())
    texts = [item["embedding_text"] for item in catalog]
    ids = [item["id"] for item in catalog]

    # Resume support: load whatever was already embedded in a prior run.
    done_vectors = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
        done_vectors = {k: v for k, v in zip(checkpoint["ids"], checkpoint["vectors"])}
        print(f"Resuming: {len(done_vectors)} items already embedded from a previous run.")

    remaining_idx = [i for i, cid in enumerate(ids) if cid not in done_vectors]
    print(f"{len(remaining_idx)} of {len(ids)} items left to embed.")

    for start in range(0, len(remaining_idx), BATCH_SIZE):
        chunk_idx = remaining_idx[start : start + BATCH_SIZE]
        batch_texts = [texts[i] for i in chunk_idx]
        vectors = embed_batch_with_retry(batch_texts)
        for i, vec in zip(chunk_idx, vectors):
            done_vectors[ids[i]] = vec
        print(f"Embedded {len(done_vectors)}/{len(texts)}")

        # checkpoint after every batch so a crash never loses more than one batch
        CHECKPOINT_PATH.write_text(
            json.dumps({"ids": list(done_vectors.keys()), "vectors": list(done_vectors.values())})
        )
        if start + BATCH_SIZE < len(remaining_idx):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # Assemble final array in the catalog's original order.
    vectors_in_order = [done_vectors[cid] for cid in ids]
    arr = np.array(vectors_in_order, dtype=np.float32)
    np.save(EMB_PATH, arr)
    IDS_PATH.write_text(json.dumps(ids))
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f"Saved {arr.shape} embeddings to {EMB_PATH}")


if __name__ == "__main__":
    main()