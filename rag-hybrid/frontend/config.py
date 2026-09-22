"""Frontend configuration. Kept independent from the backend package: only BACKEND_URL matters here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://localhost:8000"


def _from_dotenv(key: str) -> str | None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def backend_url() -> str:
    """BACKEND_URL from the process environment, then from .env, then the local default."""
    return (os.getenv("BACKEND_URL") or _from_dotenv("BACKEND_URL") or DEFAULT_URL).rstrip("/")
