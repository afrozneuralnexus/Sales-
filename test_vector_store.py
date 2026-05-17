"""Tests for VectorStore."""

import pytest
from src.vector_store import VectorStore
from src.document_processor import DocumentChunk


def make_chunks(texts, source="test.csv"):
    return [DocumentChunk(text=t, source=source) for t in texts]


@pytest.fixture
def store_with_sales_data():
    vs = VectorStore()
    chunks = make_chunks([
        "Q1 revenue was 1.2 million dollars, driven by product Widget A and Widget B.",
        "Sales team North region achieved 120% of quota in March.",
        "Customer acquisition cost dropped by 15% in Q2 due to new marketing strategy.",
        "Top performing sales rep Alice closed 45 deals worth 500K total.",
        "Churn rate increased to 8% in Q3, primarily in SMB segment.",
    ])
    vs.add_documents(chunks)
    return vs


def test_add_documents(store_with_sales_data):
    assert len(store_with_sales_data) == 5


def test_similarity_search_returns_results(store_with_sales_data):
    results = store_with_sales_data.similarity_search("revenue Q1", k=3)
    assert len(results) <= 3
    assert all(isinstance(r[0], DocumentChunk) for r in results)
    assert all(isinstance(r[1], float) for r in results)


def test_similarity_search_relevance(store_with_sales_data):
    results = store_with_sales_data.similarity_search("churn rate SMB", k=5)
    top_chunk = results[0][0].text
    assert "churn" in top_chunk.lower() or "smb" in top_chunk.lower()


def test_similarity_scores_range(store_with_sales_data):
    results = store_with_sales_data.similarity_search("sales rep performance", k=5)
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_empty_store():
    vs = VectorStore()
    results = vs.similarity_search("revenue", k=3)
    assert results == []


def test_top_k_limit(store_with_sales_data):
    results = store_with_sales_data.similarity_search("sales", k=2)
    assert len(results) <= 2


def test_no_zero_score_results(store_with_sales_data):
    results = store_with_sales_data.similarity_search("completely irrelevant zzz xyz", k=5)
    # All returned results should have score > 0 (store filters them)
    assert all(score > 0.0 for _, score in results)
