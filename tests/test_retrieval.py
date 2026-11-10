"""Unit tests for DocumentProcessor and RetrievalEngine."""

import pytest
from app.config import settings
from app.retrieval import RetrievalEngine
from app.document_processor import DocumentProcessor, chunk_text


def test_chunking_utility():
    sample = "word " * 500
    chunks = chunk_text(sample, target_tokens=300, overlap_tokens=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= 300


def test_retrieval_engine_initialization():
    engine = RetrievalEngine(settings.PAPERS_DIR, settings.PROCESSED_DIR)
    engine.initialize()
    assert engine.is_initialized
    stats = engine.get_corpus_stats()
    assert stats["total_documents"] == 3
    assert stats["config_A_passages"] > 100
    assert stats["config_B_passages"] > 100


def test_retrieval_settings_a_b_c():
    engine = RetrievalEngine(settings.PAPERS_DIR, settings.PROCESSED_DIR)
    engine.initialize()

    query = "What is the sampling rate of the Mimi neural audio codec in Moshi?"

    # Setting A: top 3
    cits_a, score_a = engine.retrieve(query, retrieval_setting="A")
    assert len(cits_a) == 3
    assert score_a > 0.10
    assert any("Moshi" in c.paper_title for c in cits_a)

    # Setting B: top 5
    cits_b, score_b = engine.retrieve(query, retrieval_setting="B")
    assert len(cits_b) == 5
    assert any("Moshi" in c.paper_title for c in cits_b)

    # Setting C: top 5 with reranking
    cits_c, score_c = engine.retrieve(query, retrieval_setting="C")
    assert len(cits_c) == 5
    assert any("Moshi" in c.paper_title for c in cits_c)


def test_retrieval_out_of_scope_query():
    engine = RetrievalEngine(settings.PAPERS_DIR, settings.PROCESSED_DIR)
    engine.initialize()

    # Query with no relevance in Transformer/Moshi/Dynamic Chunking papers
    irrelevant_query = "What is the battery life of Apple Vision Pro in spatial computing?"
    cits, max_score = engine.retrieve(irrelevant_query, retrieval_setting="B")
    # Low score triggers refusal in pipeline
    assert max_score < 0.20
