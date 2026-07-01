import json
import logging
import os
import re
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger("llm")

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        _configured = True


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction in case the model wraps the object in
    markdown fences or adds stray text despite instructions."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


DEFAULT_EXTRACTION_STATE = {
    "intent": "need_clarification",
    "role_or_job_title": None,
    "seniority": None,
    "skills_or_topics": [],
    "industry_or_context": None,
    "languages": [],
    "test_types_wanted": [],
    "max_duration_minutes": None,
    "other_constraints": [],
    "exclude_names": [],
    "compare_targets": [],
    "missing_info": ["what role or context this assessment is for"],
    "user_signals_satisfaction": False,
    "include_personality_complement": False,
    "search_query": "",
}


def call_extraction(system_prompt: str, conversation_text: str) -> dict:
    """Structured-state extraction call. Falls back to a safe default
    (ask a generic clarifying question) on any failure so the endpoint
    never breaks the response schema."""
    try:
        _ensure_configured()
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=system_prompt)
        response = model.generate_content(
            conversation_text,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                max_output_tokens=800,
            ),
            request_options={"timeout": 12},
        )
        parsed = _extract_json(response.text)
        if parsed is None:
            logger.warning("Extraction returned unparseable JSON: %r", response.text[:300])
            return dict(DEFAULT_EXTRACTION_STATE)
        # merge over defaults so missing keys never crash downstream code
        merged = dict(DEFAULT_EXTRACTION_STATE)
        merged.update({k: v for k, v in parsed.items() if v is not None})
        return merged
    except Exception:
        logger.exception("Extraction call failed; using safe default state")
        return dict(DEFAULT_EXTRACTION_STATE)


def call_generation(prompt: str, fallback: str) -> str:
    """Plain-text generation call with a guaranteed fallback string."""
    try:
        _ensure_configured()
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=400,
            ),
            request_options={"timeout": 12},
        )
        text = (response.text or "").strip()
        return text if text else fallback
    except Exception:
        logger.exception("Generation call failed; using fallback text")
        return fallback
