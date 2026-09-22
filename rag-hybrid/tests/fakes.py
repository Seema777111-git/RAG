"""Deterministic test doubles: no network, no model downloads."""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from backend.app.graph.text import content_tokens

_TRIPLE = re.compile(r"(?P<s>[A-Z]\w*(?: [A-Z]\w*)*) (?P<p>acquired|founded|makes|hired) (?P<o>[A-Z]\w*(?: [A-Z]\w*)*)")


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar wording => high cosine similarity."""

    name = "fake-embedder"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in content_tokens(text):
            v[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    """Stands in for the real LLM client. Behaviour is chosen from the system prompt."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def _respond(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if "extract knowledge-graph triples" in system:
            triples = [{"subject": m["s"], "predicate": m["p"], "object": m["o"]} for m in _TRIPLE.finditer(user)]
            return json.dumps({"triples": triples})
        if "write evaluation questions" in system:
            return json.dumps({"question": "Who acquired Widget Inc?", "answer": "Acme Corp acquired Widget Inc."})
        if "<context>" in user:
            refs = "[C1]" + ("[K1]" if "KNOWLEDGE GRAPH FACTS" in user else "")
            return f"Based on the documents, the answer is in the retrieved evidence {refs}."
        return "ok"

    async def complete(self, *, system, user, model=None, max_tokens=None, temperature=None) -> str:
        return self._respond(system, user)

    def complete_sync(self, *, system, user, model=None, max_tokens=None, temperature=None) -> str:
        return self._respond(system, user)


def guardrail_answer(text: str) -> str:
    """Yes/No decision a guardrail LLM would give for a NeMo self-check prompt."""
    answer = "No"
    m = re.search(r'user message: "(.*?)"\n', text, re.S | re.I)
    if m and "ignore the above" in m.group(1).lower():
        answer = "Yes"  # jailbreak detected -> block
    if "is the hypothesis grounded" in text.lower():
        hypothesis = text.lower().split("hypothesis:")[-1]
        answer = "No" if "moon is made of cheese" in hypothesis else "Yes"
    return answer


class FakeGuardrailChat(BaseChatModel):
    """LangChain chat model that plays the role of the guardrail LLM (Yes/No answers)."""

    @property
    def _llm_type(self) -> str:
        return "fake-guardrail"

    def _generate(self, messages: list[BaseMessage], stop: Optional[list] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=guardrail_answer(str(messages[-1].content))))])


def make_pdf(path, pages: list[str]) -> None:
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


SAMPLE_PAGES = [
    "Acme Corp acquired Widget Inc in 2021. The acquisition gave Acme Corp a strong position in the gadget market.\n\n"
    "Widget Inc makes Smart Gadgets for households across Europe and Asia.",
    "Jane Doe founded Nimbus Labs in 2015. Nimbus Labs makes Cloud Sensors used in agriculture.\n\n"
    "The annual report shows revenue growth of twelve percent driven by sensor sales.",
]
