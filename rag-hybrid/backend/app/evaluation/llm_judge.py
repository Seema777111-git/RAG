"""Adapter that lets DeepEval use our configured LLM (OpenAI or Anthropic) as its judge model."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from backend.app.core.llm import LLM

_SYSTEM = "You are a precise evaluation assistant. Follow the output format instructions exactly."


def _extract_json(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"Judge did not return JSON: {text[:200]!r}")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start : end + 1])


class LLMJudge(DeepEvalBaseLLM):
    """Implements DeepEval's model interface on top of our `LLM` protocol, so one API key serves the whole app."""

    def __init__(self, llm: LLM, model: str):
        self._llm = llm
        self._model_name = model
        super().__init__(model)

    def load_model(self, *args, **kwargs):  # DeepEval calls this from __init__
        return self

    def get_model_name(self, *args, **kwargs) -> str:
        return self._model_name

    # ------------------------------------------------------------------ prompt shaping
    @staticmethod
    def _prompt(prompt: str, schema: Optional[type[BaseModel]]) -> str:
        if schema is None:
            return prompt
        return (
            f"{prompt}\n\nReturn ONLY a JSON object that conforms to this JSON schema, with no prose and no code fences:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )

    @staticmethod
    def _parse(text: str, schema: Optional[type[BaseModel]]):
        if schema is None:
            return text
        return schema.model_validate(_extract_json(text))

    # ------------------------------------------------------------------ DeepEval interface
    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None, **_: Any):
        text = self._llm.complete_sync(system=_SYSTEM, user=self._prompt(prompt, schema), model=self._model_name, max_tokens=4096, temperature=0.0)
        return self._parse(text, schema)

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None, **_: Any):
        text = await self._llm.complete(system=_SYSTEM, user=self._prompt(prompt, schema), model=self._model_name, max_tokens=4096, temperature=0.0)
        return self._parse(text, schema)
