"""Logging setup."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
