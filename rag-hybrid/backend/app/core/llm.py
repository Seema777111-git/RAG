"""Thin async/sync wrappers around the OpenAI and Anthropic chat APIs.

Every service depends on the `LLM` protocol, never on an SDK directly. That keeps the rest of the code
easy to unit-test (inject a fake) and lets one setting (`LLM_PROVIDER`) switch the whole application.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from backend.app.core.config import Settings
from backend.app.core.errors import LLMConfigError, LLMError, RAGError

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# OpenAI reasoning models reject a custom temperature.
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def supports_temperature(model: str) -> bool:
    return not model.lower().startswith(_NO_TEMPERATURE_PREFIXES)


@runtime_checkable
class LLM(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str: ...

    def complete_sync(
        self,
        *,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str: ...


def translate_llm_error(exc: Exception, provider: str, model: str, base_url: str = "") -> LLMError:
    """Turn an SDK exception into a short message that says what to do about it."""
    name = exc.__class__.__name__
    status = getattr(exc, "status_code", None)
    detail = str(getattr(exc, "message", "") or exc).strip()[:300]
    key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        msg = f"{provider} rejected the request (HTTP {status}). Check {key_var} in .env: it must be a valid key, and its project must have access to model '{model}'. Detail: {detail}"
    elif name == "RateLimitError" or status == 429:
        msg = f"{provider} rate limit or no credit (HTTP 429). If the detail says 'insufficient_quota', add credit in your {provider} billing page; otherwise lower KG_EXTRACTION_CONCURRENCY. Detail: {detail}"
    elif name == "NotFoundError" or status == 404:
        msg = f"{provider} could not find model '{model}', or your key cannot use it. Set LLM_MODEL / EXTRACTION_MODEL / GUARDRAIL_MODEL / JUDGE_MODEL in .env to a model you can access. Detail: {detail}"
    elif name in ("APIConnectionError", "APITimeoutError"):
        cause = exc.__cause__ or exc
        target = base_url or "the default endpoint"
        msg = f"Cannot reach the {provider} API at {target}. Check your internet, VPN or proxy, and that OPENAI_BASE_URL is empty unless you really use a gateway. Cause: {str(cause)[:200]}"
    else:
        msg = f"{provider} request failed ({name}): {detail}"
    return LLMError(msg)


class _BaseLLM:
    """Shared plumbing: lazy client creation (so the API can boot without a key) and default parameters."""

    def __init__(self, settings: Settings, async_client: Any = None, sync_client: Any = None):
        self._settings = settings
        self._async_client = async_client
        self._sync_client = sync_client

    def _client_kwargs(self) -> dict:
        s = self._settings
        if not s.llm_configured:
            raise LLMConfigError(f"{s.api_key_env} is not set. Add it to your .env file and restart the API.")
        return {"api_key": s.llm_api_key, "timeout": s.llm_timeout_seconds, "max_retries": s.llm_max_retries}

    def _defaults(self, model, max_tokens, temperature) -> tuple[str, int, float]:
        s = self._settings
        return model or s.llm_model, max_tokens or s.llm_max_tokens, s.llm_temperature if temperature is None else temperature


class OpenAILLM(_BaseLLM):
    """OpenAI Chat Completions."""

    def _kwargs(self) -> dict:
        kwargs = self._client_kwargs()
        # Always explicit: never let an empty OPENAI_BASE_URL in the environment reach the SDK.
        kwargs["base_url"] = self._settings.openai_base_url.strip() or OPENAI_DEFAULT_BASE_URL
        return kwargs

    def _get_async(self):
        if self._async_client is None:
            from langsmith.wrappers import wrap_openai
            from openai import AsyncOpenAI

            self._async_client = wrap_openai(AsyncOpenAI(**self._kwargs()))
        return self._async_client

    def _get_sync(self):
        if self._sync_client is None:
            from langsmith.wrappers import wrap_openai
            from openai import OpenAI

            self._sync_client = wrap_openai(OpenAI(**self._kwargs()))
        return self._sync_client

    def _params(self, system, user, model, max_tokens, temperature) -> dict:
        model, max_tokens, temperature = self._defaults(model, max_tokens, temperature)
        params: dict = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if supports_temperature(model):
            params["temperature"] = temperature
        return params

    @staticmethod
    def _text(response) -> str:
        return (response.choices[0].message.content or "").strip()

    def _fail(self, exc: Exception, params: dict) -> LLMError:
        base = self._settings.openai_base_url.strip() or OPENAI_DEFAULT_BASE_URL
        return translate_llm_error(exc, "openai", params["model"], base)

    async def complete(self, *, system, user, model=None, max_tokens=None, temperature=None) -> str:
        params = self._params(system, user, model, max_tokens, temperature)
        try:
            return self._text(await self._get_async().chat.completions.create(**params))
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises many types; all become one readable error
            raise self._fail(exc, params) from exc

    def complete_sync(self, *, system, user, model=None, max_tokens=None, temperature=None) -> str:
        params = self._params(system, user, model, max_tokens, temperature)
        try:
            return self._text(self._get_sync().chat.completions.create(**params))
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._fail(exc, params) from exc


class AnthropicLLM(_BaseLLM):
    """Claude via the Anthropic Messages API."""

    def _get_async(self):
        if self._async_client is None:
            from anthropic import AsyncAnthropic
            from langsmith.wrappers import wrap_anthropic

            self._async_client = wrap_anthropic(AsyncAnthropic(**self._client_kwargs()))
        return self._async_client

    def _get_sync(self):
        if self._sync_client is None:
            from anthropic import Anthropic
            from langsmith.wrappers import wrap_anthropic

            self._sync_client = wrap_anthropic(Anthropic(**self._client_kwargs()))
        return self._sync_client

    def _params(self, system, user, model, max_tokens, temperature) -> dict:
        model, max_tokens, temperature = self._defaults(model, max_tokens, temperature)
        return {"model": model, "max_tokens": max_tokens, "temperature": temperature, "system": system, "messages": [{"role": "user", "content": user}]}

    @staticmethod
    def _text(response) -> str:
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()

    async def complete(self, *, system, user, model=None, max_tokens=None, temperature=None) -> str:
        params = self._params(system, user, model, max_tokens, temperature)
        try:
            return self._text(await self._get_async().messages.create(**params))
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise translate_llm_error(exc, "anthropic", params["model"]) from exc

    def complete_sync(self, *, system, user, model=None, max_tokens=None, temperature=None) -> str:
        params = self._params(system, user, model, max_tokens, temperature)
        try:
            return self._text(self._get_sync().messages.create(**params))
        except RAGError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise translate_llm_error(exc, "anthropic", params["model"]) from exc


def create_llm(settings: Settings) -> LLM:
    """Build the LLM client for the provider chosen by `LLM_PROVIDER`."""
    return OpenAILLM(settings) if settings.llm_provider == "openai" else AnthropicLLM(settings)
