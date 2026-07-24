"""
input_guard.py — Input validation and sanitisation for incoming queries.

Checks (in order):
  1. Query length    — block extremely long queries (DoS protection)
  2. Prompt injection— detect known injection patterns
  3. PII detection   — flag SSNs, credit cards, etc. in queries
  4. Empty/trivial   — reject empty or whitespace-only queries

Returns a dict:
  {
    "blocked": bool,
    "flags": List[str],     # triggered check names
    "message": str,         # message to show the user if blocked
    "sanitised_query": str, # cleaned query (for non-blocked passes)
  }

Design principle: prefer logging + flagging over silent blocking.
Only hard-block on prompt injection and extreme length.
"""

import logging
import os
import re
from typing import List

from app.guardrails.prompt_armor import detect_injection

logger = logging.getLogger(__name__)

_ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
_MAX_QUERY_LENGTH  = 2000

# ── PII Patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS = {
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone_us":    re.compile(r"\b(?:\+1\s?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# ── Profanity / Hate Speech keywords (minimal, extend as needed) ──────────────
# We only flag — not block — on these
_SENSITIVE_KEYWORDS = [
    "hack", "exploit", "bypass", "jailbreak", "sql injection",
    "xss", "csrf", "malware", "ransomware",
]


async def check_input(query: str) -> dict:
    """
    Validate and sanitise the incoming user query.
    Returns a result dict (see module docstring).
    """
    flags: List[str] = []

    if not _ENABLE_GUARDRAILS:
        return {"blocked": False, "flags": [], "message": "", "sanitised_query": query}

    # ── 1. Empty query ─────────────────────────────────────────────────────────
    stripped = query.strip()
    if not stripped:
        return {
            "blocked": True,
            "flags": ["empty_query"],
            "message": "Please enter a question.",
            "sanitised_query": "",
        }

    # ── 2. Length check ────────────────────────────────────────────────────────
    if len(stripped) > _MAX_QUERY_LENGTH:
        logger.warning("Input guard: query too long (%d chars)", len(stripped))
        return {
            "blocked": True,
            "flags": ["query_too_long"],
            "message": f"Query is too long (max {_MAX_QUERY_LENGTH} characters). Please shorten it.",
            "sanitised_query": stripped[:_MAX_QUERY_LENGTH],
        }

    # ── 3. Prompt injection detection ─────────────────────────────────────────
    injection_matches = detect_injection(stripped)
    if injection_matches:
        logger.warning("Input guard: prompt injection detected — %s", injection_matches)
        return {
            "blocked": True,
            "flags": ["prompt_injection"] + injection_matches,
            "message": "This query appears to contain instructions that override system behaviour and cannot be processed.",
            "sanitised_query": stripped,
        }

    # ── 4. PII detection (flag, don't block) ──────────────────────────────────
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(stripped):
            flags.append(f"pii_{pii_type}")
            logger.info("Input guard: PII detected in query (%s)", pii_type)

    # ── 5. Sensitive keyword flagging (flag only) ──────────────────────────────
    lower = stripped.lower()
    for kw in _SENSITIVE_KEYWORDS:
        if kw in lower:
            flags.append(f"sensitive_keyword:{kw}")

    # ── 6. Basic sanitisation ──────────────────────────────────────────────────
    # Strip leading/trailing whitespace, normalise internal whitespace
    sanitised = " ".join(stripped.split())

    return {
        "blocked": False,
        "flags": flags,
        "message": "",
        "sanitised_query": sanitised,
    }
