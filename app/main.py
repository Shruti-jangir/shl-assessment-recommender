import logging
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, RecommendationItem, HealthResponse, Message
from app.retriever import CatalogRetriever
from app.llm import call_extraction, call_generation
from app.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    CLARIFY_GENERATION_PROMPT,
    RECOMMEND_GENERATION_PROMPT,
    COMPARE_GENERATION_PROMPT,
    REFUSAL_TEMPLATES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="SHL Assessment Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = CatalogRetriever()

MAX_CLARIFY_ROUNDS = 1  # after this many clarifying turns, force a shortlist.
# Lowered from 2 after real behavior-probe testing showed the LLM's own
# intent classification can be overly conservative about moving to
# "ready_to_recommend" even once role + a real differentiator (seniority,
# duration, language) are already established — this deterministic
# backstop guarantees the agent doesn't clarify indefinitely.
DEFAULT_TOP_K = 10  # grading metric is explicitly Recall@10; use the full
                     # ceiling the spec allows so a relevant item is never
                     # truncated out purely for lack of room
                     
def format_history(messages: List[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def count_prior_clarify_rounds(messages: List[Message]) -> int:
    # crude but effective: count assistant turns so far that didn't end in a
    # question mark ... actually simpler: count assistant turns overall,
    # since we only ever ask a clarifying question before the first shortlist.
    return sum(1 for m in messages if m.role == "assistant")


def build_recommendation_items(items: List[dict]) -> List[RecommendationItem]:
    out = []
    for it in items:
        test_type = it["test_types"][0] if it["test_types"] else ""
        out.append(RecommendationItem(name=it["name"], url=it["url"], test_type=test_type))
    return out


def requirements_summary(state: dict) -> str:
    keys = [
        "role_or_job_title", "seniority", "skills_or_topics", "industry_or_context",
        "languages", "test_types_wanted", "max_duration_minutes", "other_constraints",
        "exclude_names",
    ]
    parts = []
    for k in keys:
        v = state.get(k)
        if v:
            parts.append(f"{k}: {v}")
    return "; ".join(parts) if parts else "none stated yet"


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = req.messages
    if not messages:
        return ChatResponse(
            reply="Hi! Tell me a bit about the role you're hiring for and I can help "
            "you find the right SHL assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    history_text = format_history(messages)

    try:
        state = call_extraction(EXTRACTION_SYSTEM_PROMPT, history_text)
    except Exception:
        logger.exception("Unexpected extraction failure")
        return ChatResponse(
            reply="Sorry, could you rephrase that? I want to make sure I understand the "
            "role and what you need before recommending assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    intent = state.get("intent", "need_clarification")
    prior_clarify_rounds = count_prior_clarify_rounds(messages)
    force_recommend = prior_clarify_rounds >= MAX_CLARIFY_ROUNDS

    # ---- out-of-scope / refusal paths -----------------------------------
    # Refusals are never overridden by the turn-cap forcing logic below —
    # "we're almost out of turns" is not a reason to answer a legal or
    # off-topic question.
    if intent in REFUSAL_TEMPLATES:
        return ChatResponse(
            reply=REFUSAL_TEMPLATES[intent],
            recommendations=[],
            end_of_conversation=False,
        )

    # ---- compare -----------------------------------------------------
    if intent == "compare":
        targets = state.get("compare_targets") or []
        resolved = [retriever.find_by_name(t) for t in targets if t]
        resolved = [r for r in resolved if r]
        if len(resolved) >= 2:
            items_text = "\n\n".join(
                f"### {r['name']} ({', '.join(r['test_types'])})\n"
                f"URL: {r['url']}\nDuration: {r['duration_raw'] or 'not specified'}\n"
                f"Description: {r['description']}"
                for r in resolved
            )
            prompt = COMPARE_GENERATION_PROMPT.format(items=items_text, history=history_text)
            fallback = (
                f"Here's what I have on {resolved[0]['name']} and {resolved[1]['name']}: "
                f"{resolved[0]['description'][:200]} ... {resolved[1]['description'][:200]}"
            )
            reply = call_generation(prompt, fallback)
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
        # couldn't resolve both -> ask which ones they mean
        return ChatResponse(
            reply="Which two assessments would you like compared? Could you give me the "
            "exact names?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ---- still need more context --------------------------------------
    if intent == "need_clarification" and not force_recommend:
        missing = state.get("missing_info") or ["what role or context this is for"]
        prompt = CLARIFY_GENERATION_PROMPT.format(missing_info="; ".join(missing), history=history_text)
        fallback = "Could you tell me a bit more about the role and what matters most for this hire?"
        reply = call_generation(prompt, fallback)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # ---- recommend / refine --------------------------------------------
    # Combine the LLM's synthesized query, the raw accumulated requirements,
    # AND the full raw user-turn text (not just the LLM's synthesis) — this
    # is a safety net: if extraction under-captures a turn (e.g. a short
    # confirmation message like "go with the hybrid" dilutes the query),
    # the original role/skill language from earlier turns still anchors
    # retrieval instead of being lost.
    synth_query = state.get("search_query") or ""
    req_summary = requirements_summary(state)
    all_user_text = " ".join(m.content for m in messages if m.role == "user")
    query = f"{synth_query} {req_summary} {all_user_text}".strip()
    if not query:
        query = messages[-1].content

    results = retriever.search(
        query=query,
        top_k=DEFAULT_TOP_K,
        test_types=state.get("test_types_wanted") or None,
        max_duration_minutes=state.get("max_duration_minutes"),
        exclude_names=state.get("exclude_names") or None,
    )
    
    # SHL's flagship personality assessment is conventionally bundled into
    # shortlists for people-facing roles (leadership, stakeholders, sales,
    # client-facing, cross-team collaboration) but its own catalog text is
    # generic enough that pure embedding similarity structurally can't
    # compete against narrowly-worded technical items for those queries.
    # Pin it explicitly when the extraction step flags people-facing signal,
    # rather than leaving it to a ranking contest it's built to lose.
    # SHL's flagship personality assessment (OPQ32r) is conventionally
    # bundled into shortlists for most professional/office/technical roles,
    # but NOT for narrowly operational frontline roles (which instead get
    # a sector-specific personality-type item that's already surfaced by
    # normal retrieval, e.g. "Manufac. & Indust. - Safety & Dependability").
    # Determined deterministically from keywords rather than trusting an
    # LLM boolean buried in a large JSON schema, which was dropped
    # unreliably in testing. Checks the raw conversation text too (not
    # just extracted fields) as a safety net against extraction gaps.
    NEGATIVE_PERSONALITY_KEYWORDS = [
        "contact center", "contact centre", "call center", "call centre",
        "phone support", "manufactur", "industrial", "plant operator",
        "chemical facility", "warehouse", "assembly line",
        "safety-critical", "safety critical", "retail cashier",
    ]
    combined_context = " ".join(filter(None, [
        state.get("role_or_job_title") or "",
        state.get("industry_or_context") or "",
        " ".join(state.get("other_constraints") or []),
        " ".join(state.get("skills_or_topics") or []),
        history_text,
    ])).lower()
    should_consider_opq = not any(neg in combined_context for neg in NEGATIVE_PERSONALITY_KEYWORDS)

    exclude_lower = {n.lower() for n in (state.get("exclude_names") or [])}
    if (
        should_consider_opq
        and "occupational personality questionnaire opq32r" not in exclude_lower
    ):
        opq = retriever.by_name_lower.get("occupational personality questionnaire opq32r")
        if opq and opq["id"] not in {r["id"] for r in results}:
            if len(results) >= DEFAULT_TOP_K:
                results = results[:-1]
            results.append(opq)

    if not results:
        return ChatResponse(
            reply="I couldn't find a good match in the catalog for that combination — "
            "could you tell me a bit more about the role, or relax one of the constraints?",
            recommendations=[],
            end_of_conversation=False,
        )

    rec_items = build_recommendation_items(results)
    shortlist_text = "\n".join(f"- {r['name']} ({', '.join(r['test_types'])}): {r['url']}" for r in results)
    prompt = RECOMMEND_GENERATION_PROMPT.format(
        requirements=requirements_summary(state),
        shortlist=shortlist_text,
        history=history_text,
    )
    fallback = f"Here are {len(results)} assessments from the catalog that match what you've described."
    reply = call_generation(prompt, fallback)

    end_of_conversation = bool(state.get("user_signals_satisfaction", False))

    return ChatResponse(
        reply=reply,
        recommendations=rec_items,
        end_of_conversation=end_of_conversation,
    )
