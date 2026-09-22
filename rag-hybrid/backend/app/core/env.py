"""Process-environment hygiene.

Some libraries (DeepEval, for one) copy the project's `.env` into `os.environ` when imported. A line such as
`OPENAI_BASE_URL=` then becomes an *empty* variable, and SDKs treat an empty string as a real value (an empty
server address) instead of "not set". Removing empty provider variables makes them behave as unset again.
"""

from __future__ import annotations

import os

_PREFIXES = ("OPENAI_", "ANTHROPIC_", "LANGSMITH_", "LANGCHAIN_")


def sanitize_environment() -> list[str]:
    """Delete provider-related environment variables that are set but empty. Returns the removed names."""
    removed = [name for name, value in os.environ.items() if name.startswith(_PREFIXES) and not value.strip()]
    for name in removed:
        del os.environ[name]
    return removed
