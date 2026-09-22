from __future__ import annotations

import pytest

from backend.app.core.config import Settings
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        llm_provider="openai",
        openai_api_key="test-key",
        guardrails_enabled=False,
        langsmith_tracing=False,
        chunk_size=300,
        chunk_overlap=40,
        min_chunk_chars=20,
        kg_semantic_threshold=0.5,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
