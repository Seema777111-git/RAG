import pytest

from backend.app.domain.models import Triple
from backend.app.graph.extractor import TripleExtractor, parse_triples_json
from backend.app.graph.kg_store import KnowledgeGraphStore
from backend.app.graph.linker import EntityLinker
from backend.app.graph.retriever import KnowledgeGraphRetriever
from backend.app.graph.text import normalize_entity
from tests.fakes import FakeEmbedder, FakeLLM


def T(s, p, o, chunk="c1", doc="d1"):
    return Triple(subject=s, predicate=p, object=o, chunk_id=chunk, doc_id=doc)


@pytest.fixture
def store(tmp_path):
    kg = KnowledgeGraphStore(tmp_path / "kg.graphml")
    kg.add_triples(
        [
            T("Acme Corp", "acquired", "Widget Inc", "c1"),
            T("Widget Inc", "makes", "Smart Gadgets", "c2"),
            T("Smart Gadgets", "sold in", "Europe", "c3"),
            T("Jane Doe", "founded", "Nimbus Labs", "c4", "d2"),
        ]
    )
    return kg


def test_normalize_entity():
    assert normalize_entity("  The Acme Corp. ") == "acme corp"
    assert normalize_entity("“Widget”") == "widget"


def test_duplicates_merge_chunk_ids(store):
    store.add_triples([T("acme corp", "Acquired", "the Widget Inc", "c9")])
    assert store.node_count == 6 and store.edge_count == 4
    edge = store.graph.get_edge_data("acme corp", "widget inc", key="d1::acquired")
    assert edge["chunk_ids"] == ["c1", "c9"]


def test_self_loops_and_empty_are_ignored(tmp_path):
    kg = KnowledgeGraphStore(tmp_path / "g.graphml")
    assert kg.add_triples([T("A", "is", "A"), T("", "x", "B"), T("A", "", "B")]) == 0


def test_graphml_round_trip(store, tmp_path):
    store.save()
    loaded = KnowledgeGraphStore(tmp_path / "kg.graphml")
    assert (loaded.node_count, loaded.edge_count) == (store.node_count, store.edge_count)
    edge = loaded.graph.get_edge_data("acme corp", "widget inc", key="d1::acquired")
    assert edge["chunk_ids"] == ["c1"] and edge["predicate"] == "acquired"
    assert loaded.label("acme corp") == "Acme Corp"


def test_remove_document_prunes_orphans(store):
    assert store.remove_document("d2") == 1
    assert "jane doe" not in store.graph and "nimbus labs" not in store.graph
    assert store.node_count == 4


def test_linker_exact_and_partial(store):
    linker = EntityLinker(store, None, semantic=False)
    exact = linker.link("Who did Acme Corp buy?")
    assert exact[0].label == "Acme Corp" and exact[0].method == "exact" and exact[0].score == 1.0
    partial = linker.link("Tell me about smart devices and gadgets")
    assert any(e.node_id == "smart gadgets" and e.method == "partial" for e in partial)
    assert linker.link("What is the weather?") == []


def test_linker_semantic(store):
    linker = EntityLinker(store, FakeEmbedder(), semantic=True, semantic_threshold=0.5)
    found = linker.link("europe")
    assert any(e.node_id == "europe" for e in found)


def test_retriever_hops(store):
    linker = EntityLinker(store, None, semantic=False)
    kg = KnowledgeGraphRetriever(store, linker)
    one = kg.retrieve("What about Acme Corp?", hops=1)
    assert {t.object for t in one.triples} == {"Widget Inc"}
    two = kg.retrieve("What about Acme Corp?", hops=2)
    assert {"Widget Inc", "Smart Gadgets"} <= {t.object for t in two.triples}
    assert all(n.is_seed for n in one.nodes if n.id == "acme corp")
    assert one.graph_nodes_total == store.node_count


def test_retriever_no_entities_returns_empty(store):
    kg = KnowledgeGraphRetriever(store, EntityLinker(store, None, semantic=False))
    result = kg.retrieve("completely unrelated question")
    assert result.triples == [] and result.graph_edges_total == store.edge_count


def test_parse_triples_json_variants():
    good = '```json\n{"triples": [{"subject": "A", "predicate": "b", "object": "C"}]}\n```'
    assert parse_triples_json(good)[0]["subject"] == "A"
    assert parse_triples_json("no json here") == []
    assert parse_triples_json('{"triples": "oops"}') == []
    assert parse_triples_json("Sure! " + good)[0]["object"] == "C"


async def test_extractor_dedupes_and_counts_failures():
    from backend.app.domain.models import Chunk

    llm = FakeLLM()
    ex = TripleExtractor(llm, "m", max_triples_per_chunk=5, concurrency=2)
    ok = Chunk(chunk_id="c1", doc_id="d", source="s", page=1, text="Acme Corp acquired Widget Inc. Acme Corp acquired Widget Inc.")

    class Boom(FakeLLM):
        async def complete(self, **kw):
            raise RuntimeError("api down")

    bad_ex = TripleExtractor(Boom(), "m")
    triples, failed = await ex.extract_all([ok])
    assert failed == 0 and len(triples) == 1 and triples[0].chunk_id == "c1"
    triples, failed = await bad_ex.extract_all([ok, ok])
    assert triples == [] and failed == 2
