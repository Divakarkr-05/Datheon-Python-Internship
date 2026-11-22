"""
Unit tests for the Secure Local File Reader Tool.
Tests TXT extraction, PDF parsing, boundary enforcement, error handling, and path traversal prevention.
"""

from pathlib import Path
import pytest
from app.tools.file_reader import file_reader, read_local_file


class TestFileReaderOperations:
    """Test safe file reading operations."""

    def test_read_valid_txt_file(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("sample.txt", base_dir=temp_data_dir)
        assert "File: sample.txt" in result
        assert "Datheon AI revenue" in result
        assert "TXT" in result

    def test_read_text_based_pdf_file(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("sample.pdf", base_dir=temp_data_dir)
        assert "File: sample.pdf" in result
        assert "PDF" in result
        assert "page(s)" in result
        assert "characters extracted" in result
        assert "Datheon AI revenue" in result

    def test_read_empty_pdf_notice(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("empty.pdf", base_dir=temp_data_dir)
        assert "Notice:" in result
        assert "is empty or contains no extractable text" in result

    def test_read_scanned_image_pdf_notice(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("scanned.pdf", base_dir=temp_data_dir)
        assert "Notice:" in result
        assert "contains image data" in result
        assert "text extraction is unavailable without OCR" in result

    def test_empty_file_warning(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("empty.txt", base_dir=temp_data_dir)
        assert "Warning:" in result
        assert "empty (0 bytes)" in result

    def test_missing_file_error(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("nonexistent_file.txt", base_dir=temp_data_dir)
        assert "Error: File 'nonexistent_file.txt' not found" in result

    def test_directory_target_error(self, temp_data_dir: Path, mock_config) -> None:
        sub_dir = temp_data_dir / "subdir"
        sub_dir.mkdir()
        result = read_local_file("subdir", base_dir=temp_data_dir)
        assert "is a directory" in result

    def test_unsupported_extension_error(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("document.docx", base_dir=temp_data_dir)
        assert "Error: Unsupported file format '.docx'" in result

    def test_corrupted_pdf_error(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("corrupt.pdf", base_dir=temp_data_dir)
        assert "Error:" in result
        assert "corrupt" in result.lower() or "failed" in result.lower() or "invalid" in result.lower()

    def test_langchain_tool_wrapper(self, temp_data_dir: Path, mock_config) -> None:
        result = file_reader.invoke({"filename": "sample.txt"})
        assert "sample.txt" in result
        assert file_reader.name == "file_reader"


class TestFileReaderSecurity:
    """Test path traversal prevention and sandboxing."""

    def test_path_traversal_relative_parent(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("../../etc/passwd", base_dir=temp_data_dir)
        assert "Security Error: Access denied" in result

    def test_path_traversal_windows_root(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("C:/Windows/System32/calc.exe", base_dir=temp_data_dir)
        assert "Security Error: Access denied" in result or "not found" in result

    def test_empty_filename_error(self, temp_data_dir: Path, mock_config) -> None:
        result = read_local_file("", base_dir=temp_data_dir)
        assert "Error: No filename provided" in result
