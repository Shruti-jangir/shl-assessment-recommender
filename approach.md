# Approach — SHL Conversational Assessment Recommender

## Design choices

**Stateless-by-reconstruction.** Since `/chat` carries no server-side
session, every call re-derives "what do we know so far" from the full
message history via a single structured-extraction LLM call. This call
returns a JSON object: an `intent` classification (off_topic /
legal_or_general_advice / prompt_injection / compare / need_clarification /
ready_to_recommend) plus accumulated requirements (role, seniority, skills,
languages, test types, duration limit, excluded items). Requirements
persist and accumulate turn over turn unless the user removes or
contradicts them, which is how "refine" ("actually, add personality
tests" / "drop the OPQ32r") works without any server-side state — refine
is just recommend-with-updated-requirements.

**Retrieval is deterministic; the LLM only narrates.** The biggest
correctness risk in a system like this is a hallucinated URL or an item
that isn't actually in the catalog. So the LLM never picks which
assessments appear in `recommendations` — a plain cosine-similarity
search over precomputed Gemini embeddings of all 377 catalog items
(`text-embedding-004`, embedded from name + description + category +
job levels) does that, with hard metadata filters (test type, max
duration, excluded names) layered on top. The LLM's only job at that
point is to write 2-4 sentences explaining *why* the already-decided
shortlist fits — it's given the finalized list as context, not asked to
generate it. This makes "every URL from the scraped catalog" a
structural guarantee rather than a hope.

**Refusals are templated, not LLM-generated.** Off-topic requests, legal/
compliance questions, and prompt-injection attempts get a deterministic
canned response after intent classification — no second LLM call, so
there's no chance of the refusal itself leaking into a recommendation or
being talked out of its boundary by a follow-up message. Refusals are
never overridden by the turn-cap-forcing logic (see below): running low
on turns is not a reason to answer a legal question.

**Turn-cap awareness.** The evaluator caps conversations at 8 total
turns. After 2 clarifying rounds, the agent is forced past
`need_clarification` into producing a shortlist even with a sparse
requirements set, rather than risk clarifying forever and never scoring
any recall. This trades a small amount of "ask one more good question"
polish for a strong guarantee that a shortlist actually gets produced
within the turn budget.

## Retrieval setup

- Catalog preprocessing (`scripts/preprocess_catalog.py`) normalizes the
  377 raw catalog entries (all `status: ok`, no dedup needed — none
  found), maps each item's category labels to SHL's standard letter
  codes (A/B/C/D/E/K/P/S), and builds one embedding-ready text blob per
  item.
- Embeddings are precomputed once offline (`scripts/embed_catalog.py`)
  and committed to the repo, so the deployed service never re-embeds the
  catalog at startup — only the live query gets embedded per turn (one
  fast API call), keeping cold starts and per-turn latency low and
  quota usage minimal.
- Search combines cosine similarity (85%) with a light keyword-overlap
  score (15%) so exact terms (e.g. a named technology like "Spring")
  don't get diluted by pure semantic similarity, then applies hard
  filters for requested test type / max duration / excluded names —
  filters are only applied if they don't zero out the candidate set
  entirely, so an imperfect inferred constraint can't silently kill all
  results.

## Prompt design

Two LLM calls per turn, max: one JSON-mode extraction/classification
call, and one plain-text narration call (skipped entirely for
refusals). Both have explicit timeouts (12s) and safe fallbacks — if
either call fails or returns unparseable output, the endpoint still
returns a valid, schema-compliant response (a generic clarifying
question, or a plain factual shortlist summary) instead of a 500. This
was a deliberate defense against "code that works on the happy path and
breaks on anything else."

## Evaluation approach

- `scripts/smoke_test.py`: offline, mocks the LLM layer, verifies FastAPI
  routing/schema compliance and the "no hallucinated URLs" guarantee for
  every intent branch without needing network access. Run on every
  change before touching the real API.
- `scripts/eval_traces.py`: replays each of the 10 provided gold traces'
  user turns against a live instance of the service and computes
  Recall@10 of the final shortlist against each trace's expected final
  table, plus hard-eval checks (schema fields present, URLs genuinely in
  catalog, recommendation count in [1,10]).

Mean Recall@10 across the 10 public traces: 0.510 (up from an initial 0.253
before tuning — see iteration notes below).

Traces that failed hard evals: none — all 10 traces returned schema-valid
responses with recommendation counts in [1,10] and URLs that were
verifiably from the scraped catalog on every turn.

Traces with recall < 1.0 and why: most residual misses are near-duplicate
product-family confusion rather than wrong-domain retrieval — e.g.
recommending "Verify - G+" instead of "SHL Verify Interactive G+", or a
generic OPQ report variant instead of the specific "OPQ Leadership Report"
the trace expected. The catalog has many closely related variants of the
same underlying assessment (different report formats, "(New)" vs legacy
versions, sector-specific bundles), and pure embedding similarity doesn't
always separate them cleanly. Two iterations meaningfully improved recall:
(1) discovering that ~70% of gold traces expect SHL's flagship personality
assessment (OPQ32r) bundled into the shortlist for any people-facing role,
which the embedding-only ranker structurally couldn't surface against
narrowly-worded technical queries — fixed with a deterministic keyword
rule rather than an LLM-flagged field, which proved unreliable when buried
in a large JSON extraction schema; (2) widening the retrieval query to
include full raw user-turn text as a safety net against thin per-turn
LLM query synthesis. Chose to stop further precision tuning at this point
to avoid overfitting retrieval specifically to these 10 public traces at
the expense of the private holdout set.

## What didn't work / trade-offs

- An earlier version let the extraction call directly propose catalog
  item names for the shortlist; this risked near-miss names that don't
  exactly match a catalog entry (spelling, "(New)" suffixes, etc.) and
  broke the "no hallucination" guarantee. Switching to
  embedding-retrieval-picks/LLM-only-narrates fixed this structurally.
- Considered per-conversation caching/session state for speed, but the
  spec is explicit that the service is stateless and stores no
  per-conversation state, so state is fully reconstructed every call by
  design, not as a workaround.
- The `google-generativeai` SDK is deprecated in favor of `google-genai`;
  kept the older SDK given the time constraint since it still functions
  correctly (only a warning), and a blind migration this close to the
  deadline without a way to test real API calls in the build environment
  was a worse risk than the deprecation itself. A production version
  should migrate.

## AI tool usage disclosure

Built with Claude (agentic coding assistance) for scaffolding the
FastAPI service, retriever, prompt design, and the offline smoke-test
suite. The design decisions above (deterministic retrieval vs.
LLM-picked items, refusal templating, turn-cap forcing, stateless
reconstruction) were reasoned through and can be defended/discussed in
the technical interview.
