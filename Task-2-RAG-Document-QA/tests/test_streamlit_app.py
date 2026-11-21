"""
Unit and integration tests for the Streamlit application and helper functions.
All tests run offline using mocks without calling external APIs or downloading models.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from streamlit_app import (
    validate_uploaded_file,
    save_uploaded_file,
    process_uploaded_documents,
    submit_question,
    clear_chat_history,
    reset_vector_database,
    init_session_state,
    filter_chunks,
)


class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self.size = len(content)

    def getbuffer(self):
        return self._content


# 1. File Upload Validation
def test_validate_uploaded_file_valid():
    valid_pdf = MockUploadedFile("contract.pdf", b"%PDF-1.4 mock content")
    valid, err = validate_uploaded_file(valid_pdf)
    assert valid is True
    assert err is None

    valid_txt = MockUploadedFile("notes.txt", b"Some textual data")
    valid, err = validate_uploaded_file(valid_txt)
    assert valid is True
    assert err is None


def test_validate_uploaded_file_invalid_extension():
    bad_file = MockUploadedFile("data.csv", b"col1,col2\nval1,val2")
    valid, err = validate_uploaded_file(bad_file)
    assert valid is False
    assert "Unsupported file type" in err


def test_validate_uploaded_file_empty():
    empty_file = MockUploadedFile("empty.txt", b"")
    valid, err = validate_uploaded_file(empty_file)
    assert valid is False
    assert "empty (0 bytes)" in err


def test_validate_uploaded_file_oversized():
    huge_file = MockUploadedFile("huge.pdf", b"x" * (25 * 1024 * 1024))
    valid, err = validate_uploaded_file(huge_file)
    assert valid is False
    assert "exceeds the maximum allowed limit of 20 MB" in err


def test_validate_uploaded_file_none():
    valid, err = validate_uploaded_file(None)
    assert valid is False
    assert "No file provided" in err


# 2. Save Uploaded File
def test_save_uploaded_file(tmp_path: Path):
    mock_file = MockUploadedFile("test_report.pdf", b"test binary stream")
    saved_path = save_uploaded_file(mock_file, target_dir=tmp_path)

    assert saved_path.exists()
    assert saved_path.name == "test_report.pdf"
    assert saved_path.read_bytes() == b"test binary stream"


# 3. Document Processing Workflow (Mocked)
def test_process_uploaded_documents_empty():
    res = process_uploaded_documents([])
    assert res["status"] == "error"
    assert "No files selected" in res["message"]


def test_process_uploaded_documents_success(tmp_path: Path):
    mock_file = MockUploadedFile("policy.txt", b"The refund window is thirty days.")

    with patch("streamlit_app.DATA_DIR", tmp_path):
        with patch("streamlit_app.get_vector_store") as mock_get_vs:
            with patch("streamlit_app.add_documents_to_vector_store") as mock_add:
                mock_vs = MagicMock()
                mock_get_vs.return_value = mock_vs
                mock_add.return_value = ["id1"]

                res = process_uploaded_documents([mock_file], persist_directory=tmp_path)
                assert res["status"] == "success"
                assert res["doc_count"] == 1
                assert res["chunk_count"] > 0
                assert "policy.txt" in res["files"]


# 4. Question Submission & Error Handling
def test_submit_question_empty():
    res = submit_question("   ")
    assert "Please enter a valid question" in res["answer"]
    assert res["citations"] == []


def test_submit_question_missing_api_key():
    with patch("streamlit_app.get_google_api_key", side_effect=ValueError("Key missing")):
        res = submit_question("What is the policy?")
        assert "API Key Configuration Error" in res["answer"]


def test_submit_question_success():
    mock_response = {
        "query": "What is the policy?",
        "answer": "The refund window is 30 days. [Source: policy.txt, Page: 1]",
        "citations": ["policy.txt (Page 1)"],
        "source_documents": [],
    }

    with patch("streamlit_app.get_google_api_key", return_value="mock_key"):
        with patch("streamlit_app.answer_question", return_value=mock_response):
            res = submit_question("What is the policy?")
            assert res["answer"] == mock_response["answer"]
            assert res["citations"] == ["policy.txt (Page 1)"]


# 5. Session State & Chat History Management
def test_session_state_and_clear_chat():
    with patch("streamlit_app.st") as mock_st:
        mock_st.session_state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "indexed_files": ["manual.pdf"],
            "total_chunks": 5,
        }

        # Clear chat should only clear messages
        clear_chat_history()
        assert mock_st.session_state["messages"] == []
        # Vector database metadata must remain intact!
        assert mock_st.session_state["indexed_files"] == ["manual.pdf"]
        assert mock_st.session_state["total_chunks"] == 5


# 6. Database Reset Behavior
def test_reset_vector_database(tmp_path: Path):
    dummy_doc = tmp_path / "sample.txt"
    dummy_doc.write_text("dummy")

    with patch("streamlit_app.DATA_DIR", tmp_path):
        with patch("streamlit_app.clear_vector_store", return_value=True) as mock_clear:
            with patch("streamlit_app.st") as mock_st:
                mock_st.session_state = {
                    "messages": [{"role": "user", "content": "Question"}],
                    "indexed_files": ["sample.txt"],
                    "total_chunks": 12,
                }

                success = reset_vector_database(persist_directory=tmp_path)
                assert success is True
                mock_clear.assert_called_once_with(persist_directory=tmp_path)
                assert mock_st.session_state["indexed_files"] == []
                assert mock_st.session_state["total_chunks"] == 0
                # Chat messages should NOT be cleared by database reset
                assert len(mock_st.session_state["messages"]) == 1


# 7. Chunk Explorer Filter Helper Tests
def test_filter_chunks_by_source():
    sample_chunks = [
        {"id": "1", "source": "guide.pdf", "content": "Installing the system."},
        {"id": "2", "source": "invoice.txt", "content": "Total amount due."},
        {"id": "3", "source": "guide.pdf", "content": "Configuring the database."},
    ]

    # Filter specific source
    guide_chunks = filter_chunks(sample_chunks, source_filter="guide.pdf")
    assert len(guide_chunks) == 2
    assert all(c["source"] == "guide.pdf" for c in guide_chunks)

    # Filter All Sources
    all_chunks = filter_chunks(sample_chunks, source_filter="All Sources")
    assert len(all_chunks) == 3


def test_filter_chunks_by_search_query():
    sample_chunks = [
        {"id": "1", "source": "guide.pdf", "content": "Installing the database cluster."},
        {"id": "2", "source": "invoice.txt", "content": "Total payment amount due."},
    ]

    matched = filter_chunks(sample_chunks, search_query="database")
    assert len(matched) == 1
    assert matched[0]["id"] == "1"

    # Case-insensitive
    matched_upper = filter_chunks(sample_chunks, search_query="PAYMENT")
    assert len(matched_upper) == 1
    assert matched_upper[0]["id"] == "2"

    # No match
    no_match = filter_chunks(sample_chunks, search_query="nonexistent")
    assert len(no_match) == 0


def test_submit_question_includes_retrieved_context():
    mock_response = {
        "query": "Where is the headquarters?",
        "answer": "San Francisco, CA.",
        "citations": ["company.pdf (Page 1)"],
        "source_documents": [],
        "retrieved_context": [
            {"rank": 1, "source": "company.pdf", "page": 1, "content": "HQ is located in SF.", "score": 0.15}
        ],
    }

    with patch("streamlit_app.get_google_api_key", return_value="mock_key"):
        with patch("streamlit_app.answer_question", return_value=mock_response):
            res = submit_question("Where is the headquarters?")
            assert "retrieved_context" in res
            assert len(res["retrieved_context"]) == 1
            assert res["retrieved_context"][0]["rank"] == 1
            assert res["retrieved_context"][0]["score"] == 0.15
