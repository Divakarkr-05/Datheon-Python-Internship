"""
Tests for ChromaDB vector store operations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.vector_store import (
    get_vector_store,
    add_documents_to_vector_store,
    similarity_search,
    clear_vector_store,
    get_stored_chunks,
    get_stored_chunks_summary,
)


def test_vector_store_operations(tmp_path: Path):
    mock_embeddings = MagicMock()

    with patch("app.vector_store.Chroma") as mock_chroma_class:
        mock_store = MagicMock()
        mock_chroma_class.return_value = mock_store
        mock_store.add_documents.return_value = ["id1", "id2"]
        mock_store.similarity_search.return_value = [
            Document(page_content="chunk 1", metadata={"source": "doc.pdf", "page": 1})
        ]

        # 1. Initialize store
        store = get_vector_store(persist_directory=tmp_path, embedding_function=mock_embeddings)
        assert store == mock_store

        # 2. Add documents
        docs = [
            Document(page_content="chunk 1", metadata={"source": "doc.pdf", "page": 1}),
            Document(page_content="chunk 2", metadata={"source": "doc.pdf", "page": 2}),
        ]
        ids = add_documents_to_vector_store(store, docs)
        assert ids == ["id1", "id2"]

        # 3. Add empty documents
        empty_ids = add_documents_to_vector_store(store, [])
        assert empty_ids == []

        # 4. Similarity search
        results = similarity_search(store, query="test query", k=1)
        assert len(results) == 1
        assert results[0].page_content == "chunk 1"

        # 5. Empty query similarity search
        empty_results = similarity_search(store, query="  ")
        assert empty_results == []


def test_clear_vector_store(tmp_path: Path):
    dummy_file = tmp_path / "chroma.sqlite3"
    dummy_file.write_text("data")
    dummy_dir = tmp_path / "index"
    dummy_dir.mkdir()
    (dummy_dir / "index.bin").write_text("index data")

    with patch("app.vector_store.Chroma"):
        with patch("app.vector_store.get_embedding_model"):
            cleared = clear_vector_store(persist_directory=tmp_path)
            assert cleared is True
            assert not dummy_file.exists()
            assert not dummy_dir.exists()


def test_get_stored_chunks_success():
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_col.get.return_value = {
        "ids": ["id-1", "id-2"],
        "documents": ["First paragraph text", "Second paragraph text"],
        "metadatas": [
            {"source": "report.pdf", "page": 1, "chunk_index": 0},
            {"source": "report.pdf", "page": 2, "chunk_index": 1},
        ],
    }

    chunks = get_stored_chunks(vector_store=mock_store)
    assert len(chunks) == 2
    assert chunks[0]["id"] == "id-1"
    assert chunks[0]["source"] == "report.pdf"
    assert chunks[0]["page"] == 1
    assert chunks[0]["char_count"] == len("First paragraph text")
    assert chunks[1]["id"] == "id-2"


def test_get_stored_chunks_empty():
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    chunks = get_stored_chunks(vector_store=mock_store)
    assert chunks == []


def test_get_stored_chunks_source_filter():
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_col.get.return_value = {
        "ids": ["id-1"],
        "documents": ["Filtered content"],
        "metadatas": [{"source": "manual.pdf", "page": 3}],
    }

    chunks = get_stored_chunks(vector_store=mock_store, source_filter="manual.pdf")
    mock_col.get.assert_called_once_with(
        include=["documents", "metadatas"],
        where={"source": "manual.pdf"},
    )
    assert len(chunks) == 1
    assert chunks[0]["source"] == "manual.pdf"


def test_get_stored_chunks_pagination():
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_col.get.return_value = {
        "ids": ["id-3"],
        "documents": ["Page 2 content"],
        "metadatas": [{"source": "notes.txt", "page": 1}],
    }

    chunks = get_stored_chunks(vector_store=mock_store, limit=10, offset=20)
    mock_col.get.assert_called_once_with(
        include=["documents", "metadatas"],
        limit=10,
        offset=20,
    )
    assert len(chunks) == 1


def test_get_stored_chunks_summary():
    mock_store = MagicMock()
    mock_col = MagicMock()
    mock_store._collection = mock_col
    mock_col.get.return_value = {
        "ids": ["id-1", "id-2", "id-3"],
        "documents": ["Doc 1 text", "Doc 2 text", "Doc 1 text continued"],
        "metadatas": [
            {"source": "alpha.pdf", "page": 1},
            {"source": "beta.txt", "page": 1},
            {"source": "alpha.pdf", "page": 2},
        ],
    }

    summary = get_stored_chunks_summary(vector_store=mock_store)
    assert summary["total_chunks"] == 3
    assert summary["total_documents"] == 2
    assert summary["sources"] == ["alpha.pdf", "beta.txt"]
    assert summary["counts_per_source"] == {"alpha.pdf": 2, "beta.txt": 1}
