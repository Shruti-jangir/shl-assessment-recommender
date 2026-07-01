import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import google.generativeai as genai

logger = logging.getLogger("retriever")

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog_processed.json"
EMB_PATH = ROOT / "data" / "catalog_embeddings.npy"
IDS_PATH = ROOT / "data" / "catalog_embeddings_ids.json"

EMBED_MODEL = "models/gemini-embedding-001"


class CatalogRetriever:
    """In-memory semantic + keyword retriever over the SHL catalog.

    Deterministic by design: the LLM never invents which items appear in
    `recommendations` — it only narrates a shortlist that THIS class
    already picked from the real catalog. That's what guarantees every
    URL returned is genuinely from the scraped catalog (a hard eval).
    """

    def __init__(self):
        self.catalog = json.loads(CATALOG_PATH.read_text())
        self.by_id = {item["id"]: item for item in self.catalog}
        self.by_name_lower = {item["name"].lower(): item for item in self.catalog}

        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        if EMB_PATH.exists() and IDS_PATH.exists():
            self.embeddings = np.load(EMB_PATH)
            self.ids = json.loads(IDS_PATH.read_text())
            # normalize once for fast cosine similarity via dot product
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            self._unit_embeddings = self.embeddings / norms
        else:
            logger.warning(
                "No cached catalog embeddings found at %s — semantic search "
                "will fall back to keyword matching only. Run "
                "scripts/embed_catalog.py to enable it.",
                EMB_PATH,
            )
            self._unit_embeddings = None

    # ------------------------------------------------------------------
    def _embed_query(self, text: str) -> Optional[np.ndarray]:
        try:
            result = genai.embed_content(
                model=EMBED_MODEL, content=text, task_type="retrieval_query"
            )
            vec = np.array(result["embedding"], dtype=np.float32)
            n = np.linalg.norm(vec)
            return vec / n if n > 0 else vec
        except Exception:
            logger.exception("Query embedding failed; falling back to keyword search")
            return None

    def _keyword_score(self, item: dict, query: str) -> float:
        q_terms = {t for t in query.lower().split() if len(t) > 2}
        if not q_terms:
            return 0.0
        haystack = (item["name"] + " " + item["description"]).lower()
        hits = sum(1 for t in q_terms if t in haystack)
        return hits / max(len(q_terms), 1)

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        test_types: Optional[List[str]] = None,
        max_duration_minutes: Optional[int] = None,
        exclude_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Returns up to top_k catalog items ranked by relevance to `query`,
        honoring optional hard filters."""
        exclude_lower = {n.lower() for n in (exclude_names or [])}

        candidates = self.catalog
        if test_types:
            wanted = set(test_types)
            filtered = [c for c in candidates if wanted & set(c["test_types"])]
            # Only apply the filter if it doesn't wipe out everything —
            # a slightly-wrong inferred type shouldn't zero out results.
            if filtered:
                candidates = filtered
        if max_duration_minutes:
            filtered = [
                c
                for c in candidates
                if c["duration_minutes"] is None or c["duration_minutes"] <= max_duration_minutes
            ]
            if filtered:
                candidates = filtered
        if exclude_lower:
            candidates = [c for c in candidates if c["name"].lower() not in exclude_lower]

        query_vec = self._embed_query(query) if self._unit_embeddings is not None and query.strip() else None

        scored = []
        id_to_idx = {cid: i for i, cid in enumerate(self.ids)} if self._unit_embeddings is not None else {}
        for item in candidates:
            sem_score = 0.0
            if query_vec is not None and item["id"] in id_to_idx:
                sem_score = float(np.dot(self._unit_embeddings[id_to_idx[item["id"]]], query_vec))
            kw_score = self._keyword_score(item, query)
            # semantic carries the concept match; keyword weight raised so
            # exact product/technology names (e.g. "OPQ32r", "Excel") reliably
            # outrank near-duplicate family variants that are semantically
            # similar but not the specific thing asked for
            combined = (0.7 * sem_score) + (0.3 * kw_score)
            scored.append((combined, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def find_by_name(self, name: str) -> Optional[dict]:
        """Fuzzy-ish lookup used for /compare — tries exact match, then
        substring match, then falls back to semantic search top-1."""
        name_l = name.lower().strip()
        if name_l in self.by_name_lower:
            return self.by_name_lower[name_l]
        for item in self.catalog:
            if name_l in item["name"].lower() or item["name"].lower() in name_l:
                return item
        results = self.search(name, top_k=1)
        return results[0] if results else None
