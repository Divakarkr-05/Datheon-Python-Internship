"""
Tests for document chunking, parameter validation, and metadata preservation.
"""

import pytest
from langchain_core.documents import Document

from app.chunking import chunk_documents, validate_chunking_parameters


def test_chunking_metadata_preservation():
    docs = [
        Document(
            page_content="A" * 1500,
            metadata={"source": "test_guide.pdf", "page": 3},
        )
    ]

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "test_guide.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["chunk_index"] == idx


def test_chunking_empty_input():
    assert chunk_documents([]) == []


def test_chunking_empty_document_contents():
    docs = [Document(page_content="   ", metadata={"source": "blank.txt"})]
    assert chunk_documents(docs) == []


def test_validate_chunking_parameters():
    # Valid
    validate_chunking_parameters(1000, 200)

    # Invalid chunk_size
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        validate_chunking_parameters(0, 200)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        validate_chunking_parameters(-50, 10)

    # Invalid chunk_overlap
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        validate_chunking_parameters(500, -1)

    # Overlap >= size
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        validate_chunking_parameters(500, 500)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        validate_chunking_parameters(500, 600)
