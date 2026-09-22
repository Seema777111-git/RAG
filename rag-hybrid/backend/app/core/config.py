"""Application settings, loaded from environment variables and the project `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# (strong model for answers and judging, fast/cheap model for extraction and guardrails)
PROVIDER_MODELS = {
    "openai": ("gpt-4o", "gpt-4o-mini"),
    "anthropic": ("claude-sonnet-5", "claude-haiku-4-5-20251001"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ LLM provider
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = ""  # optional: Azure, a proxy, or another OpenAI-compatible gateway
    anthropic_api_key: SecretStr = SecretStr("")
    # Leave blank to use the provider's default (see PROVIDER_MODELS).
    llm_model: str = ""  # answer generation
    extraction_model: str = ""  # knowledge-graph triple extraction (many cheap calls)
    guardrail_model: str = ""  # NeMo Guardrails self-check rails
    judge_model: str = ""  # DeepEval judge
    llm_max_tokens: int = Field(1024, ge=64)
    llm_temperature: float = Field(0.0, ge=0.0, le=1.0)
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # ------------------------------------------------------------------ Embeddings / chunking
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    chunk_size: int = Field(900, ge=200)  # characters
    chunk_overlap: int = Field(150, ge=0)
    min_chunk_chars: int = 40

    # ------------------------------------------------------------------ Retrieval
    vector_top_k: int = Field(5, ge=1, le=50)
    kg_hops: int = Field(1, ge=1, le=3)
    kg_max_triples: int = Field(15, ge=1, le=100)
    kg_max_entities: int = Field(8, ge=1, le=30)
    kg_semantic_linking: bool = True
    kg_semantic_threshold: float = Field(0.55, ge=0.0, le=1.0)
    kg_chunk_boost: float = 0.10  # score bonus for chunks that both retrievers agree on
    kg_only_chunks: int = 3  # max extra passages pulled in because a KG triple cites them
    context_max_chars: int = 12000

    # ------------------------------------------------------------------ Knowledge-graph extraction
    kg_extraction_enabled: bool = True
    kg_max_triples_per_chunk: int = 12
    kg_extraction_concurrency: int = Field(4, ge=1, le=32)

    # ------------------------------------------------------------------ Guardrails (NVIDIA NeMo)
    guardrails_enabled: bool = True
    guardrails_fail_open: bool = False  # if a rail itself errors: allow (True) or block (False)

    # ------------------------------------------------------------------ Evaluation (DeepEval)
    eval_threshold: float = Field(0.5, ge=0.0, le=1.0)
    eval_concurrency: int = Field(2, ge=1, le=16)

    # ------------------------------------------------------------------ LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "rag-hybrid"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ------------------------------------------------------------------ API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    max_upload_mb: int = 50
    log_level: str = "INFO"

    # ------------------------------------------------------------------ Storage
    data_dir: Path = PROJECT_ROOT / "data"

    # ------------------------------------------------------------------ Frontend
    backend_url: str = "http://localhost:8000"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _fill_model_defaults(self) -> Settings:
        strong, fast = PROVIDER_MODELS[self.llm_provider]
        self.llm_model = self.llm_model.strip() or strong
        self.judge_model = self.judge_model.strip() or strong
        self.extraction_model = self.extraction_model.strip() or fast
        self.guardrail_model = self.guardrail_model.strip() or fast
        return self

    @field_validator("data_dir", mode="before")
    @classmethod
    def _resolve_data_dir(cls, value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    # ------------------------------------------------------------------ Derived values
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def vector_dir(self) -> Path:
        return self.indexes_dir / "faiss"

    @property
    def graph_path(self) -> Path:
        return self.indexes_dir / "knowledge_graph.graphml"

    @property
    def registry_path(self) -> Path:
        return self.indexes_dir / "documents.json"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_key_env(self) -> str:
        """Name of the environment variable that holds the active provider's key."""
        return "OPENAI_API_KEY" if self.llm_provider == "openai" else "ANTHROPIC_API_KEY"

    @property
    def llm_api_key(self) -> str:
        secret = self.openai_api_key if self.llm_provider == "openai" else self.anthropic_api_key
        return secret.get_secret_value().strip()

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.uploads_dir, self.indexes_dir, self.vector_dir, self.eval_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
