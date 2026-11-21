"""
Tests for document loader (PDF, TXT, error cases, and empty file handling).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document

from app.document_loader import load_document, load_documents_from_directory


def test_load_txt_file(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a sample document for testing RAG.", encoding="utf-8")

    docs = load_document(txt_file)
    assert len(docs) == 1
    assert "This is a sample document" in docs[0].page_content
    assert docs[0].metadata["source"] == "sample.txt"
    assert docs[0].metadata["page"] == 1


def test_load_empty_txt_file(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    docs = load_document(empty_file)
    assert docs == []


def test_load_whitespace_txt_file(tmp_path: Path):
    ws_file = tmp_path / "whitespace.txt"
    ws_file.write_text("   \n\t   ", encoding="utf-8")

    docs = load_document(ws_file)
    assert docs == []


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_file.txt")


def test_load_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "data.docx"
    bad_file.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad_file)


@patch("app.document_loader.PyPDFLoader")
def test_load_pdf_file(mock_pdf_loader, tmp_path: Path):
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock binary")

    mock_instance = MagicMock()
    mock_instance.load.return_value = [
        Document(page_content="Page one content", metadata={"page": 0}),
        Document(page_content="Page two content", metadata={"page": 1}),
    ]
    mock_pdf_loader.return_value = mock_instance

    docs = load_document(pdf_file)
    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert docs[0].metadata["source"] == "report.pdf"


def test_load_documents_from_directory(tmp_path: Path):
    file1 = tmp_path / "doc1.txt"
    file1.write_text("Document one text", encoding="utf-8")
    file2 = tmp_path / "doc2.txt"
    file2.write_text("Document two text", encoding="utf-8")
    file3 = tmp_path / "ignored.json"
    file3.write_text("{}", encoding="utf-8")

    docs = load_documents_from_directory(tmp_path)
    assert len(docs) == 2
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"doc1.txt", "doc2.txt"}


def test_load_oversized_file(tmp_path: Path):
    oversized_file = tmp_path / "large.txt"
    oversized_file.write_text("dummy")

    real_stat = oversized_file.stat()
    with patch("pathlib.Path.stat") as mock_stat:
        mock_result = MagicMock()
        mock_result.st_mode = real_stat.st_mode
        mock_result.st_size = 25 * 1024 * 1024
        mock_stat.return_value = mock_result

        with pytest.raises(ValueError, match="exceeds the maximum allowed size of 20 MB"):
            load_document(oversized_file)

