"""
Secure Local File Reader Tool for Task 3.
Safely extracts readable text from .txt and .pdf files located inside the project's data directory.
Enforces strict path traversal barriers and handles missing, empty, or corrupt files.
"""

from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
import pypdf

from app.core.config import config
from app.core.logger import logger

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def _resolve_safe_path(filename: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve file path relative to the base data directory.
    Enforces strict isolation to prevent path traversal attacks.
    """
    target_base = (base_dir or config.data_dir).resolve()
    clean_name = filename.strip().strip("'\"").lstrip("\\/ ")

    # Resolve candidate path
    candidate = (target_base / clean_name).resolve()

    # Path traversal check
    if not candidate.is_relative_to(target_base):
        raise PermissionError(
            f"Security Error: Access denied. Path '{filename}' attempts to escape the data directory."
        )

    return candidate


def read_local_file(filename: str, base_dir: Optional[Path] = None) -> str:
    """
    Read text from a local TXT or PDF file inside the configured data directory.
    Returns the file content string or an informative error message.
    """
    if not filename or not filename.strip():
        return "Error: No filename provided to file_reader."

    try:
        resolved_path = _resolve_safe_path(filename, base_dir=base_dir)
    except PermissionError as pe:
        logger.warning("Path traversal attempt blocked: %s", filename)
        return str(pe)
    except Exception as exc:
        return f"Error: Invalid file path '{filename}': {str(exc)}"

    if not resolved_path.exists():
        return f"Error: File '{filename}' not found in data directory ({config.data_dir.name}/)."

    if not resolved_path.is_file():
        return f"Error: Path '{filename}' is a directory, not a readable file."

    # File size validation
    try:
        file_size = resolved_path.stat().st_size
    except OSError as e:
        return f"Error accessing file metadata for '{filename}': {str(e)}"

    if file_size == 0:
        return f"Warning: File '{filename}' is empty (0 bytes)."

    if file_size > config.max_file_size_bytes:
        max_mb = config.max_file_size_bytes / (1024 * 1024)
        return f"Error: File '{filename}' ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)."

    ext = resolved_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return f"Error: Unsupported file format '{ext}'. Supported formats are: {supported}."

    # Extract TXT
    if ext == ".txt":
        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
            char_count = len(content)
            return (
                f"File: {resolved_path.name} (TXT, {char_count} characters)\n"
                f"--- Content ---\n"
                f"{content.strip()}"
            )
        except Exception as exc:
            return f"Error: Failed to read text file '{filename}': {str(exc)}"

    # Extract PDF
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(str(resolved_path))
            total_pages = len(reader.pages)
            if total_pages == 0:
                return f"Notice: PDF file '{resolved_path.name}' contains 0 pages (empty document)."

            raw_page_texts = []
            has_images = False
            for page in reader.pages:
                txt = page.extract_text() or ""
                raw_page_texts.append(txt.strip())
                try:
                    if len(page.images) > 0:
                        has_images = True
                except Exception:
                    pass

            total_extracted_chars = sum(len(t) for t in raw_page_texts)

            if total_extracted_chars == 0:
                if has_images:
                    return (
                        f"Notice: PDF file '{resolved_path.name}' ({total_pages} page(s)) contains image data, "
                        "but text extraction is unavailable without OCR support. No readable text could be extracted."
                    )
                return (
                    f"Notice: PDF file '{resolved_path.name}' ({total_pages} page(s)) is empty or contains no extractable text."
                )

            extracted_pages = []
            for page_num, (page, page_text) in enumerate(zip(reader.pages, raw_page_texts), start=1):
                if page_text:
                    extracted_pages.append(f"--- Page {page_num} ---\n{page_text}")
                else:
                    try:
                        page_has_img = len(page.images) > 0
                    except Exception:
                        page_has_img = False
                    if page_has_img:
                        extracted_pages.append(
                            f"--- Page {page_num} ---\n[Image-only page; text extraction unavailable without OCR]"
                        )
                    else:
                        extracted_pages.append(f"--- Page {page_num} ---\n[Blank page]")

            full_body = "\n\n".join(extracted_pages).strip()
            return (
                f"File: {resolved_path.name} (PDF, {total_pages} page(s), {total_extracted_chars} characters extracted)\n"
                f"{full_body}"
            )
        except pypdf.errors.PdfReadError as pdf_err:
            return f"Error: PDF file '{filename}' is corrupted or invalid: {str(pdf_err)}"
        except Exception as exc:
            return f"Error: Failed to parse PDF file '{filename}': {str(exc)}"

    return f"Error: Unhandled file type '{ext}'."


@tool("file_reader")
def file_reader(filename: str) -> str:
    """
    Read the contents of a local file (.txt or .pdf) located in the project's data directory.
    Use this tool whenever the user asks about local files, company policies, internal reports, or stored data.
    Examples:
      - 'sample.txt'
      - 'company_policy.txt'
    Input must be the filename or relative path within the data directory.
    """
    return read_local_file(filename)
