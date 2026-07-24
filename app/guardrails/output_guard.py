"""
output_guard.py — Output validation and sanitisation for generated answers.

Checks:
  1. Canary token leakage — if system prompt was leaked in output
  2. Faithfulness disclaimer — adds warning when score < threshold
  3. Broken image link cleanup — removes image refs pointing to dead URLs
  4. Length sanity — truncates excessively long answers

Returns a dict:
  {
    "answer": str,         # final (possibly modified) answer
    "flags": List[str],    # triggered check names
  }
"""

import logging
import os
import re
from typing import List

from app.guardrails.prompt_armor import contains_canary

logger = logging.getLogger(__name__)

_ENABLE_GUARDRAILS   = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
_FAITH_DISCLAIMER_TH = 0.65   # Below this score → add low-confidence disclaimer
_MAX_ANSWER_LENGTH   = 8000    # Chars; truncate if above

_LOW_CONFIDENCE_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *Note: This answer may not be fully supported by the retrieved context. "
    "Please verify key claims against the original page.*"
)

_CANARY_RESPONSE = (
    "I'm sorry, but I cannot provide that information. "
    "Please ask a question about the current page content."
)

# Markdown image pattern: ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


async def check_output(answer: str, faithfulness_score: float = 1.0) -> dict:
    """
    Validate and sanitise the generated answer.
    Returns {"answer": str, "flags": List[str]}.
    """
    flags: List[str] = []

    if not _ENABLE_GUARDRAILS:
        return {"answer": answer, "flags": []}

    if not answer:
        return {"answer": answer, "flags": []}

    # ── 1. Canary token leakage ────────────────────────────────────────────────
    if contains_canary(answer):
        logger.error("OUTPUT_GUARD: Canary token detected — system prompt leakage!")
        flags.append("canary_leak")
        return {"answer": _CANARY_RESPONSE, "flags": flags}

    # ── 2. Length sanity ───────────────────────────────────────────────────────
    if len(answer) > _MAX_ANSWER_LENGTH:
        logger.warning("OUTPUT_GUARD: Answer too long (%d chars) — truncating.", len(answer))
        flags.append("answer_truncated")
        answer = answer[:_MAX_ANSWER_LENGTH] + "\n\n*[Response truncated for length.]*"

    # ── 3. Broken / data-URI image cleanup ────────────────────────────────────
    # Keep only proper http/https image URLs; remove data URIs (too large for display)
    def _clean_image(match):
        alt = match.group(1)
        url = match.group(2)
        if url.startswith("data:"):
            flags.append("data_uri_image_removed")
            return f"*[Image: {alt}]*"
        return match.group(0)  # Keep as-is

    answer = _MD_IMAGE_RE.sub(_clean_image, answer)

    # ── 4. Low-faithfulness disclaimer ────────────────────────────────────────
    if faithfulness_score < _FAITH_DISCLAIMER_TH:
        flags.append("low_faithfulness_disclaimer")
        answer = answer + _LOW_CONFIDENCE_DISCLAIMER
        logger.info(
            "OUTPUT_GUARD: Low faithfulness (%.2f) — disclaimer added.", faithfulness_score
        )

    return {"answer": answer, "flags": flags}
