"""
Tests for RAG QA chain, API key handling, grounded prompt context, and citations.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from app.qa_chain import (
    get_google_api_key,
    format_context_chunks,
    answer_question,
)


def test_get_google_api_key_missing():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google API key not found"):
            get_google_api_key()


def test_get_google_api_key_present():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaSyTestKey123"}, clear=True):
        assert get_google_api_key() == "AIzaSyTestKey123"

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyGeminiKey456"}, clear=True):
        assert get_google_api_key() == "AIzaSyGeminiKey456"


def test_format_context_chunks_empty():
    assert format_context_chunks([]) == "No relevant document excerpts found."


def test_format_context_chunks():
    docs = [
        Document(page_content="Server setup instructions.", metadata={"source": "guide.pdf", "page": 2}),
        Document(page_content="Firewall configuration.", metadata={"source": "security.txt", "page": 1}),
    ]
    formatted = format_context_chunks(docs)
    assert "[Source: guide.pdf, Page: 2]" in formatted
    assert "Server setup instructions." in formatted
    assert "[Source: security.txt, Page: 1]" in formatted
    assert "Firewall configuration." in formatted


def test_answer_question_empty_query():
    res = answer_question("")
    assert "Please provide a valid question" in res["answer"]
    assert res["citations"] == []


def test_answer_question_no_relevant_docs():
    with patch("app.qa_chain.retrieve_relevant_documents_with_scores", return_value=[]):
        res = answer_question("What is the refund policy?")
        assert "cannot find the answer" in res["answer"]
        assert res["source_documents"] == []
        assert res["retrieved_context"] == []


def test_answer_question_with_mocked_llm():
    docs = [
        Document(page_content="The warranty period is 24 months.", metadata={"source": "policy.pdf", "page": 5}),
    ]

    mock_llm = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = "The warranty period is 24 months. [Source: policy.pdf, Page: 5]"

    with patch("app.qa_chain.retrieve_relevant_documents_with_scores", return_value=[(docs[0], 0.1234)]):
        with patch("app.qa_chain.build_qa_chain", return_value=mock_runnable):
            res = answer_question("What is the warranty period?", llm=mock_llm)
            assert "The warranty period is 24 months." in res["answer"]
            assert res["citations"] == ["policy.pdf (Page 5)"]
            assert len(res["source_documents"]) == 1
            assert len(res["retrieved_context"]) == 1
            assert res["retrieved_context"][0]["rank"] == 1
            assert res["retrieved_context"][0]["score"] == 0.1234
            assert res["retrieved_context"][0]["source"] == "policy.pdf"
