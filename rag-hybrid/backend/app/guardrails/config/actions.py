"""Custom NeMo Guardrails actions (auto-loaded from the config folder)."""

from __future__ import annotations

import re
from typing import Optional

from nemoguardrails.actions import action

MAX_QUESTION_CHARS = 2000

# Deliberately conservative patterns: obvious secrets that should never be sent to an LLM.
_PII_PATTERNS = [
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # payment-card-like digit runs
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"(?i)\b(?:password|passwd|api[_ -]?key|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),  # API-key-like tokens
]


@action(name="check_input_policy")
async def check_input_policy(context: Optional[dict] = None) -> bool:
    """Return True when the user message is allowed, False when it must be blocked."""
    text = ((context or {}).get("user_message") or "").strip()
    if not text or len(text) > MAX_QUESTION_CHARS:
        return False
    return not any(p.search(text) for p in _PII_PATTERNS)
