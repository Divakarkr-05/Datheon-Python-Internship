"""
Pytest configuration and shared test fixtures for Task 3.
Ensures tests run completely offline without requiring network or external API credits.
"""

from pathlib import Path
import sys
import pytest
import pypdf
from langchain_community.chat_models.fake import FakeMessagesListChatModel

from app.core.config import AppConfig


class MockToolCallingChatModel(FakeMessagesListChatModel):
    """Mock Chat Model that supports bind_tools for testing tool-calling agents."""

    def bind_tools(self, tools, **kwargs):
        return self


def make_text_pdf_bytes(text: str) -> bytes:
    """Generate a clean, conforming PDF byte string containing readable text."""
    content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("latin1")
    stream_len = len(content)
    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")
    offsets = []

    def add_obj(obj_str: bytes) -> None:
        offsets.append(len(pdf))
        pdf.extend(obj_str)

    add_obj(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n")
    add_obj(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n")
    add_obj(b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <</Font <</F1 4 0 R>>>> /Contents 5 0 R>>\nendobj\n")
    add_obj(b"4 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n")
    add_obj(f"5 0 obj\n<</Length {stream_len}>>\nstream\n".encode("latin1") + content + b"\nendstream\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin1"))
    pdf.extend(f"trailer\n<</Size {len(offsets) + 1} /Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1"))
    return bytes(pdf)


def make_scanned_image_pdf_bytes() -> bytes:
    """Generate a PDF containing an image XObject with zero extractable text (simulating a scanned document)."""
    img_stream = b"\xff\x00\x00"
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
        b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources <</XObject <</Im1 4 0 R>>>> /Contents 5 0 R>>\nendobj\n"
        b"4 0 obj\n<</Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3>>\nstream\n"
        + img_stream
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<</Length 16>> stream\nq 1 0 0 1 0 0 cm /Im1 Do Q\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer <</Size 6 /Root 1 0 R>>\nstartxref\n0\n%%EOF\n"
    )


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary data directory with comprehensive test files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normal TXT file
    sample_txt = data_dir / "sample.txt"
    sample_txt.write_text("Datheon AI revenue in 2025 was $12,500,000.", encoding="utf-8")

    # 2. Empty TXT file
    empty_txt = data_dir / "empty.txt"
    empty_txt.write_text("", encoding="utf-8")

    # 3. Unsupported file extension
    unsupported = data_dir / "document.docx"
    unsupported.write_text("Unsupported format content", encoding="utf-8")

    # 4. Valid text-based PDF file
    sample_pdf = data_dir / "sample.pdf"
    sample_pdf.write_bytes(make_text_pdf_bytes("Datheon AI revenue in 2025 was 12.5M"))

    # 5. Empty PDF file (1 page, no text content)
    empty_pdf = data_dir / "empty.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with open(empty_pdf, "wb") as f:
        writer.write(f)

    # 6. Scanned image PDF file (has image XObject, 0 text)
    scanned_pdf = data_dir / "scanned.pdf"
    scanned_pdf.write_bytes(make_scanned_image_pdf_bytes())

    # 7. Corrupted PDF file
    corrupt_pdf = data_dir / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"NOT_A_VALID_PDF_HEADER_DATA")

    return data_dir


@pytest.fixture
def mock_config(temp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """Provide an AppConfig pointing to the isolated test directory."""
    test_cfg = AppConfig(
        data_dir=temp_data_dir,
        gemini_api_key="test_dummy_api_key_12345",
        gemini_model="gemini-3.1-flash-lite",
        search_provider="duckduckgo",
        search_max_results=3,
        max_iterations=5,
    )

    import app.core.config
    import app.tools.file_reader
    import app.tools.web_search
    import app.agent.executor
    import app.main

    monkeypatch.setattr(sys.modules["app.core.config"], "config", test_cfg)
    monkeypatch.setattr(sys.modules["app.tools.file_reader"], "config", test_cfg)
    monkeypatch.setattr(sys.modules["app.tools.web_search"], "config", test_cfg)
    monkeypatch.setattr(sys.modules["app.agent.executor"], "config", test_cfg)
    monkeypatch.setattr(sys.modules["app.main"], "config", test_cfg)

    return test_cfg
