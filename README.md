# SHL Conversational Assessment Recommender

FastAPI service implementing the take-home spec: a stateless `/chat` agent
that clarifies, recommends, refines, and compares SHL Individual Test
Solutions, grounded in the real scraped catalog.

## Architecture (one paragraph)

Every `/chat` call re-derives conversation state from the full message
history (stateless, per the spec). A Gemini call (`app/llm.py:call_extraction`)
classifies intent (off-topic / legal-advice / prompt-injection / compare /
need-clarification / ready-to-recommend) and extracts accumulated
requirements as structured JSON. Off-topic/legal/injection requests get a
deterministic canned refusal — no LLM narration risk there. For
recommend/refine, a **deterministic retriever** (`app/retriever.py`,
cosine similarity over precomputed Gemini embeddings of the 377 catalog
items + metadata filters) picks the actual shortlist — the LLM is only
used to narrate *why*, never to pick items itself, so URLs can never be
hallucinated. Compare pulls the two matched catalog entries and answers
grounded only in their scraped descriptions.

## One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Gemini API key in it, then
`export $(cat .env | xargs)` or use `python-dotenv` — or just:

```bash
export GEMINI_API_KEY=your_key_here
```

## Step 1 — Preprocess the catalog (already done, but re-run if the raw
catalog file changes)

```bash
python scripts/preprocess_catalog.py
```

## Step 2 — Precompute catalog embeddings (REQUIRED before first run)

This needs your real Gemini key and internet access — run it locally:

```bash
python scripts/embed_catalog.py
```

This writes `data/catalog_embeddings.npy` and
`data/catalog_embeddings_ids.json`. **Commit both files to git** — the
deployed service just loads them, it does not re-embed the catalog at
startup (fast cold starts, no wasted quota).

## Step 3 — Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Check it's alive:
```bash
curl http://localhost:8000/health
```

## Step 4 — Test

Offline routing/schema sanity check (mocks the LLM, no API key needed):
```bash
python scripts/smoke_test.py
```

Real end-to-end replay against the 10 gold traces (needs the server
running from Step 3, and your real API key since this hits Gemini for
real):
```bash
python scripts/eval_traces.py
```
This prints per-trace Recall@10 vs. the trace's expected final shortlist,
flags any hallucinated URLs, and reports the mean Recall@10. Traces where
recall < 1.0 print the expected vs. got sets so you can see what's missing
and adjust the retrieval query/filters.

## Step 5 — Deploy to Render

1. Push this repo to GitHub (must include `data/catalog_embeddings.npy`).
2. On Render: New → Web Service → connect the repo. Render will read
   `render.yaml` automatically (or set manually: build command
   `pip install -r requirements.txt`, start command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`).
3. Set the `GEMINI_API_KEY` environment variable in the Render dashboard
   (do NOT commit it to git).
4. After deploy, hit `https://<your-app>.onrender.com/health` once to wake
   it up (free tier cold-starts can take up to ~1 minute), then re-run
   `python scripts/eval_traces.py --url https://<your-app>.onrender.com`
   to confirm the deployed version behaves the same as local.

## Step 6 — Submit

- Public endpoint URL (the Render `.onrender.com` URL)
- `approach.md` (design write-up, ≤2 pages)

## Repo layout

```
app/
  main.py        FastAPI app, /health + /chat, decision logic
  retriever.py    Deterministic catalog search (embeddings + filters)
  llm.py          Gemini wrapper (extraction call + generation call), safe fallbacks
  prompts.py      System prompts + refusal templates
  schemas.py      Pydantic request/response models (exact API spec)
scripts/
  preprocess_catalog.py   raw catalog -> clean catalog_processed.json
  embed_catalog.py        one-time embedding precompute (run locally)
  smoke_test.py           offline routing/schema tests (mocked LLM)
  eval_traces.py          real end-to-end recall eval against gold traces
data/
  shl_product_catalog.json         raw scrape (provided)
  catalog_processed.json           cleaned/normalized (generated)
  catalog_embeddings.npy           precomputed embeddings (generated, commit this)
  catalog_embeddings_ids.json      row-id mapping for the above (generated, commit this)
  sample_conversations/            10 gold traces (provided)
```
