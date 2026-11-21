"""Utility functions for logging, JSON parsing, file I/O, and model validation."""

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import BinaryIO, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
import pypdf

T = TypeVar("T", bound=BaseModel)


def setup_logger(name: str = "doc_extractor", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a formatted console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def extract_json_string(raw_text: str) -> str:
    """
    Extract a valid JSON string from raw LLM output.
    Handles markdown fences (```json ... ```), raw JSON objects, and removes trailing junk.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned empty or whitespace-only response")

    text = raw_text.strip()

    # Case 1: Markdown code block with or without language tag
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Case 2: If the text is already a clean JSON object or array
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text

    # Case 3: Locate outer braces { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    # Case 4: Locate outer brackets [ ... ]
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1].strip()

    raise ValueError(f"No valid JSON structure found in response: '{text[:100]}...'")


def parse_and_validate(raw_text: str, model_cls: Type[T]) -> T:
    """
    Extract, parse JSON from LLM output, and validate against a target Pydantic model.

    Raises:
        ValueError: If JSON is malformed or unextractable.
        ValidationError: If JSON data fails Pydantic schema validation.
    """
    json_str = extract_json_string(raw_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON returned by LLM: {err}\nRaw snippet: {json_str[:200]}") from err

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object (dict), but got {type(data).__name__}")

    try:
        validated_instance = model_cls.model_validate(data)
        return validated_instance
    except ValidationError as err:
        raise err


MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def extract_text_from_pdf(source: Union[str, Path, bytes, BinaryIO]) -> str:
    """
    Extract all readable text from a PDF source (file path, bytes, or file-like stream).

    Args:
        source: File path (str/Path), raw PDF bytes, or file-like buffer.

    Returns:
        Extracted plain text across all pages.

    Raises:
        ValueError: If PDF is encrypted, corrupted, or has no extractable text.
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found at: {path.resolve()}")
            reader = pypdf.PdfReader(str(path))
        elif isinstance(source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(source))
        else:
            reader = pypdf.PdfReader(source)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise ValueError("PDF is password-protected and cannot be read.") from e

        text_pages = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_pages.append(page_text.strip())

        extracted = "\n\n".join(text_pages).strip()
        if not extracted:
            raise ValueError("No extractable text found in PDF document (it may be a scanned image without OCR).")

        return extracted

    except FileNotFoundError:
        raise
    except Exception as err:
        if isinstance(err, ValueError):
            raise
        raise ValueError(f"Failed to read PDF document: {err}") from err


def read_document_file(file_path: Union[str, Path]) -> str:
    """
    Read document file with UTF-8 encoding and enforce 20MB file size limit.
    Supports PDF (.pdf) and text (.txt) files.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {path.resolve()}")
    if not path.is_file():
        raise ValueError(f"Provided path is not a file: {path.resolve()}")

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB"
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in [".txt", ""]:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Source file is empty: {path.resolve()}")
        return content
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Please provide a PDF (.pdf) or text (.txt) file.")


def save_json(data: Union[BaseModel, dict], output_path: Union[str, Path]) -> str:
    """Save Pydantic model or dict to a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        dump_data = data.model_dump(mode="json")
    else:
        dump_data = data

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2, ensure_ascii=False)

    return str(path.resolve())
