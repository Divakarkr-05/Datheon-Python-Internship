"""
Document loading utilities for PDF and TXT files.
Preserves page numbers, source filenames, and metadata while handling
unreadable or empty files gracefully.
"""

from pathlib import Path
from typing import List, Union
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a single PDF or TXT file into a list of LangChain Document objects.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        List of Document objects with standardized metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Handle 0-byte empty files gracefully
    file_size = path.stat().st_size
    if file_size == 0:
        logger.warning(f"File '{path.name}' is empty (0 bytes).")
        return []

    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File '{path.name}' exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB "
            f"({file_size / (1024 * 1024):.2f} MB)."
        )

    documents: List[Document] = []

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            raw_docs = loader.load()
            for doc in raw_docs:
                content = (doc.page_content or "").strip()
                if not content:
                    continue
                # Ensure 1-based page numbering for consistency
                raw_page = doc.metadata.get("page", 0)
                page_num = raw_page + 1 if isinstance(raw_page, int) and raw_page >= 0 else 1
                doc.metadata["source"] = path.name
                doc.metadata["source_path"] = str(path)
                doc.metadata["page"] = page_num
                documents.append(doc)

        elif suffix == ".txt":
            text_content = ""
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    text_content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            content = text_content.strip()
            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                        "page": 1,
                    },
                )
                documents.append(doc)
            else:
                logger.warning(f"TXT file '{path.name}' contains no readable text.")

    except Exception as exc:
        logger.error(f"Error reading document '{path}': {exc}", exc_info=True)
        return []

    return documents


def load_documents_from_directory(directory_path: Union[str, Path]) -> List[Document]:
    """
    Load all supported documents from a directory.

    Args:
        directory_path: Path to the target directory.

    Returns:
        Aggregated list of Document objects.
    """
    dir_path = Path(directory_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"Directory '{dir_path}' does not exist or is not a directory.")
        return []

    all_docs: List[Document] = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(file_path)
                all_docs.extend(docs)
            except Exception as err:
                logger.warning(f"Skipping '{file_path.name}' due to error: {err}")

    return all_docs
