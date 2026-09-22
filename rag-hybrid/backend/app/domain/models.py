"""Pydantic models shared by every layer (ingestion, retrieval, API, UI)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- documents
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str  # original filename
    page: int  # 1-based
    text: str


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    chunk_id: str
    doc_id: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    triples: int
    size_bytes: int = 0
    ingested_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- retrieval
class VectorHit(BaseModel):
    rank: int
    score: float
    chunk: Chunk


class LinkedEntity(BaseModel):
    node_id: str
    label: str
    score: float
    method: Literal["exact", "partial", "semantic"]


class KGTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    chunk_id: str
    doc_id: str
    score: float
    hop: int
    ref: Optional[str] = None  # K1, K2 ... assigned during fusion


class GraphNode(BaseModel):
    id: str
    label: str
    is_seed: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str


class KGResult(BaseModel):
    entities: list[LinkedEntity] = Field(default_factory=list)
    triples: list[KGTriple] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    graph_nodes_total: int = 0
    graph_edges_total: int = 0


class ContextChunk(BaseModel):
    ref: str  # C1, C2 ...
    chunk: Chunk
    score: float
    origin: Literal["vector", "graph", "both"]
    in_prompt: bool = True


class ContextBundle(BaseModel):
    text: str = ""
    chunks: list[ContextChunk] = Field(default_factory=list)
    triples: list[KGTriple] = Field(default_factory=list)
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not any(c.in_prompt for c in self.chunks) and not self.triples


# --------------------------------------------------------------------------- guardrails / tracing
class GuardrailResult(BaseModel):
    stage: Literal["input", "output"]
    status: Literal["passed", "modified", "blocked", "skipped", "error"]
    rail: Optional[str] = None
    message: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.status in ("passed", "modified", "skipped")


class TraceInfo(BaseModel):
    run_id: str
    project: str


# --------------------------------------------------------------------------- query
QueryMode = Literal["hybrid", "vector", "graph"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: QueryMode = "hybrid"
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    kg_hops: Optional[int] = Field(default=None, ge=1, le=3)
    kg_max_triples: Optional[int] = Field(default=None, ge=1, le=100)


class RAGAnswer(BaseModel):
    question: str
    answer: str
    mode: QueryMode
    blocked: bool = False
    blocked_by: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    vector_hits: list[VectorHit] = Field(default_factory=list)
    kg: KGResult = Field(default_factory=KGResult)
    context: ContextBundle = Field(default_factory=ContextBundle)
    guardrails: dict[str, GuardrailResult] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    trace: Optional[TraceInfo] = None
