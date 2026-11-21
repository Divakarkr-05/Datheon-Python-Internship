"""
Tests for document retriever layer and k configuration.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from app.retriever import get_retriever, retrieve_relevant_documents, retrieve_relevant_documents_with_scores


def test_get_retriever():
    mock_store = MagicMock()
    retriever = get_retriever(vector_store=mock_store, k=5)
    mock_store.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={"k": 5}
    )


def test_get_retriever_invalid_k():
    mock_store = MagicMock()
    with pytest.raises(ValueError, match="k must be a positive integer"):
        get_retriever(vector_store=mock_store, k=0)


def test_retrieve_relevant_documents_empty_query():
    mock_store = MagicMock()
    docs = retrieve_relevant_documents(query="   ", vector_store=mock_store)
    assert docs == []


def test_retrieve_relevant_documents_invalid_k():
    mock_store = MagicMock()
    with pytest.raises(ValueError, match="k must be a positive integer"):
        retrieve_relevant_documents(query="hello", vector_store=mock_store, k=-2)


def test_retrieve_relevant_documents_empty_collection():
    mock_store = MagicMock()
    mock_store._collection = MagicMock()
    mock_store._collection.count.return_value = 0

    docs = retrieve_relevant_documents(query="test question", vector_store=mock_store)
    assert docs == []


def test_retrieve_relevant_documents_success():
    mock_store = MagicMock()
    mock_store._collection = MagicMock()
    mock_store._collection.count.return_value = 10
    expected_docs = [
        Document(page_content="Information content", metadata={"source": "manual.pdf", "page": 1})
    ]

    with patch("app.retriever.similarity_search", return_value=expected_docs):
        docs = retrieve_relevant_documents(query="test question", vector_store=mock_store, k=2)
        assert docs == expected_docs


def test_retrieve_relevant_documents_error_resilience():
    mock_store = MagicMock()
    mock_store._collection = MagicMock()
    mock_store._collection.count.return_value = 10

    with patch("app.retriever.similarity_search", side_effect=RuntimeError("Index corrupt")):
        docs = retrieve_relevant_documents(query="test question", vector_store=mock_store)
        assert docs == []


def test_retrieve_relevant_documents_with_scores():
    mock_store = MagicMock()
    mock_store._collection = MagicMock()
    mock_store._collection.count.return_value = 5

    sample_doc = Document(page_content="Matched text", metadata={"source": "faq.pdf", "page": 2})
    mock_store.similarity_search_with_score.return_value = [(sample_doc, 0.2851)]

    scored_results = retrieve_relevant_documents_with_scores(query="sample query", vector_store=mock_store, k=1)
    assert len(scored_results) == 1
    doc, score = scored_results[0]
    assert doc == sample_doc
    assert score == 0.2851
