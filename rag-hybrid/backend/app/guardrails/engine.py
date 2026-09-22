"""NVIDIA NeMo Guardrails wrapper.

NeMo is used as a *guard*, not as the dialogue manager: the RAG pipeline stays in our code and NeMo
runs the configured input rails before retrieval and the output rails (safety + grounding) after
generation, through `LLMRails.check_async`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

# NeMo Guardrails >= 0.2x routes non-OpenAI providers through LangChain only when asked to.
# This must be set before `nemoguardrails` is imported.
os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

from langsmith import traceable  # noqa: E402

from backend.app.core.config import Settings  # noqa: E402
from backend.app.core.llm import OPENAI_DEFAULT_BASE_URL, supports_temperature  # noqa: E402
from backend.app.domain.models import GuardrailResult  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"
MAX_EVIDENCE_CHARS = 8000


class GuardrailsEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_model: Any = None,
        config_dir: Path = DEFAULT_CONFIG_DIR,
        enabled: Optional[bool] = None,
    ):
        """`chat_model` is any LangChain chat model; when omitted, one is built lazily for the configured provider."""
        self._settings = settings
        self._chat_model = chat_model
        self._config_dir = Path(config_dir)
        self.enabled = settings.guardrails_enabled if enabled is None else enabled
        self._rails = None
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup
    def _build_chat_model(self) -> Any:
        """LangChain chat model for the rails, matching the configured provider."""
        s = self._settings
        if not s.llm_configured:
            raise RuntimeError(f"{s.api_key_env} is not set; guardrails cannot call the LLM.")
        common = {"model": s.guardrail_model, "api_key": s.llm_api_key, "max_tokens": 64, "timeout": s.llm_timeout_seconds, "max_retries": s.llm_max_retries}
        if supports_temperature(s.guardrail_model):
            common["temperature"] = 0.0
        if s.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            common["base_url"] = s.openai_base_url.strip() or OPENAI_DEFAULT_BASE_URL
            return ChatOpenAI(**common)
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**common)

    async def _get_rails(self):
        if self._rails is not None:
            return self._rails
        async with self._init_lock:
            if self._rails is None:
                from nemoguardrails import LLMRails, RailsConfig

                config = RailsConfig.from_path(str(self._config_dir))
                llm = self._chat_model or self._build_chat_model()
                self._rails = await asyncio.to_thread(LLMRails, config, llm)
                logger.info("NeMo Guardrails initialised from %s", self._config_dir)
        return self._rails

    # ------------------------------------------------------------------ helpers
    def _failure(self, stage: str, exc: Exception, started: float) -> GuardrailResult:
        logger.exception("Guardrail %s check failed", stage)
        status = "skipped" if self._settings.guardrails_fail_open else "error"
        return GuardrailResult(
            stage=stage,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            rail="guardrails-unavailable",
            message=f"Guardrails could not run: {exc}",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    @staticmethod
    def _to_result(stage: str, result: Any, started: float) -> GuardrailResult:
        return GuardrailResult(
            stage=stage,  # type: ignore[arg-type]
            status=result.status.value,
            rail=result.rail,
            message=result.content if result.status.value != "passed" else None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    # ------------------------------------------------------------------ public API
    @traceable(name="guardrail_input", run_type="chain")
    async def check_input(self, question: str) -> GuardrailResult:
        started = time.perf_counter()
        if not self.enabled:
            return GuardrailResult(stage="input", status="skipped")
        try:
            from nemoguardrails.rails.llm.options import RailType

            rails = await self._get_rails()
            result = await rails.check_async([{"role": "user", "content": question}], rail_types=[RailType.INPUT])
            return self._to_result("input", result, started)
        except Exception as exc:  # noqa: BLE001
            return self._failure("input", exc, started)

    @traceable(name="guardrail_output", run_type="chain")
    async def check_output(self, question: str, answer: str, evidence: list[str]) -> GuardrailResult:
        """Run the output rails: safety self-check plus a grounding check against `evidence`."""
        started = time.perf_counter()
        if not self.enabled:
            return GuardrailResult(stage="output", status="skipped")
        try:
            from nemoguardrails.rails.llm.options import RailType

            rails = await self._get_rails()
            joined = "\n\n".join(evidence)[:MAX_EVIDENCE_CHARS]
            messages = [
                # `relevant_chunks` must be a string; `check_facts` opts this message into the grounding rail.
                {"role": "context", "content": {"relevant_chunks": joined, "check_facts": bool(joined)}},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            result = await rails.check_async(messages, rail_types=[RailType.OUTPUT])
            return self._to_result("output", result, started)
        except Exception as exc:  # noqa: BLE001
            return self._failure("output", exc, started)
