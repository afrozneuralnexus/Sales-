"""Tests for RAGEngine (Grok API calls are mocked)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.rag_engine import RAGEngine
from src.vector_store import VectorStore
from src.document_processor import DocumentChunk


@pytest.fixture
def populated_store():
    vs = VectorStore()
    chunks = [
        DocumentChunk(text="Q1 total revenue was $2.5M, up 20% YoY.", source="sales.xlsx", sheet="Q1"),
        DocumentChunk(text="Top product: Widget Pro with $800K revenue.", source="sales.xlsx", sheet="Q1"),
        DocumentChunk(text="Sales rep Alice had the highest close rate at 65%.", source="report.pdf", page=3),
    ]
    vs.add_documents(chunks)
    return vs


def mock_grok_response(content: str):
    return MagicMock(
        status_code=200,
        json=lambda: {
            "choices": [{"message": {"content": content}}]
        },
        raise_for_status=lambda: None,
    )


@patch("src.rag_engine.requests.post")
def test_query_returns_answer(mock_post, populated_store):
    mock_post.return_value = mock_grok_response("Q1 revenue was $2.5M, a 20% year-over-year increase.")

    engine = RAGEngine(api_key="xai-test", vector_store=populated_store, top_k=3)
    result = engine.query("What was Q1 revenue?")

    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "sources" in result


@patch("src.rag_engine.requests.post")
def test_query_includes_sources(mock_post, populated_store):
    mock_post.return_value = mock_grok_response("Widget Pro was the top product.")

    engine = RAGEngine(api_key="xai-test", vector_store=populated_store, top_k=3)
    result = engine.query("Which product had the highest revenue?")

    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0


@patch("src.rag_engine.requests.post")
def test_query_with_chat_history(mock_post, populated_store):
    mock_post.return_value = mock_grok_response("Alice had the highest close rate at 65%.")

    history = [
        {"role": "user", "content": "Tell me about Q1 revenue"},
        {"role": "assistant", "content": "Q1 revenue was $2.5M."},
    ]
    engine = RAGEngine(api_key="xai-test", vector_store=populated_store, top_k=3)
    result = engine.query("Who was the top sales rep?", chat_history=history)

    assert result["answer"]
    # Verify history was passed to the API
    call_args = mock_post.call_args
    messages = call_args.kwargs["json"]["messages"]
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


@patch("src.rag_engine.requests.post")
def test_empty_store_returns_no_data_message(mock_post):
    vs = VectorStore()
    engine = RAGEngine(api_key="xai-test", vector_store=vs, top_k=3)
    result = engine.query("What is revenue?")

    mock_post.assert_not_called()
    assert "no relevant data" in result["answer"].lower()


@patch("src.rag_engine.requests.post")
def test_api_error_raises(mock_post, populated_store):
    mock_post.return_value = MagicMock(
        raise_for_status=MagicMock(side_effect=Exception("401 Unauthorized"))
    )
    engine = RAGEngine(api_key="xai-bad-key", vector_store=populated_store)
    with pytest.raises(Exception):
        engine.query("revenue?")
