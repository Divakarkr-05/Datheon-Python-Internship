"""
ChromaDB vector store integration using langchain-chroma.
Provides persistent storage, collection management, similarity search,
and safe reset operations.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import shutil
import logging
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent / "chroma_db"
DEFAULT_COLLECTION_NAME = "document_qa_collection"


def get_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
) -> Chroma:
    """
    Create or connect to a persistent Chroma vector store collection.

    Args:
        persist_directory: Directory for Chroma persistence.
        collection_name: Collection name.
        embedding_function: Reusable embedding instance. If None, uses default.

    Returns:
        Chroma vector store instance.
    """
    target_dir = Path(persist_directory or DEFAULT_PERSIST_DIRECTORY).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    embeddings = embedding_function or get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(target_dir),
    )
    return vector_store


def add_documents_to_vector_store(
    vector_store: Chroma,
    documents: List[Document],
) -> List[str]:
    """
    Add document chunks to the Chroma vector store.

    Args:
        vector_store: Target Chroma store.
        documents: List of Document chunks with metadata.

    Returns:
        List of added document IDs.
    """
    if not documents:
        logger.warning("No documents provided to add to vector store.")
        return []

    ids = vector_store.add_documents(documents)
    logger.info(f"Successfully indexed {len(ids)} chunks into ChromaDB.")
    return ids


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Perform similarity search on the Chroma collection.

    Args:
        vector_store: Active Chroma vector store.
        query: User search query text.
        k: Number of nearest chunks to retrieve.

    Returns:
        List of top-k Document matches.
    """
    if not query or not query.strip():
        return []

    return vector_store.similarity_search(query=query, k=k)


def get_stored_chunks(
    vector_store: Optional[Chroma] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    source_filter: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Retrieve stored document chunks, metadata, and IDs from ChromaDB.
    Runs entirely read-only without regenerating embeddings or altering data.

    Args:
        vector_store: Active Chroma instance.
        persist_directory: Directory for Chroma persistence.
        collection_name: Target collection name.
        source_filter: Optional filter by source filename.
        limit: Maximum number of chunks to return.
        offset: Pagination offset.

    Returns:
        List of chunk dictionaries with metadata and content.
    """
    store = vector_store or get_vector_store(
        persist_directory=persist_directory, collection_name=collection_name
    )

    try:
        col = getattr(store, "_collection", None)
        if col is None:
            return []

        # Prepare kwargs for Chroma collection get
        get_kwargs: Dict[str, Any] = {
            "include": ["documents", "metadatas"],
        }
        if source_filter and source_filter != "All" and source_filter != "All Sources":
            get_kwargs["where"] = {"source": source_filter}
        if limit is not None and limit > 0:
            get_kwargs["limit"] = limit
        if offset > 0:
            get_kwargs["offset"] = offset

        data = col.get(**get_kwargs)

        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []

        chunks: List[Dict[str, Any]] = []
        for i, chunk_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": meta,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 1),
                "chunk_index": meta.get("chunk_index", i),
                "char_count": len(content),
            })

        return chunks
    except Exception as exc:
        logger.error(f"Error fetching stored chunks from ChromaDB: {exc}", exc_info=True)
        return []


def get_stored_chunks_summary(
    vector_store: Optional[Chroma] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Get aggregated summary of stored documents and chunks in ChromaDB.

    Returns:
        Dict with total_chunks, total_documents, sources list, and counts_per_source dict.
    """
    chunks = get_stored_chunks(
        vector_store=vector_store,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    total_chunks = len(chunks)
    counts_per_source: Dict[str, int] = {}

    for c in chunks:
        src = c.get("source", "Unknown")
        counts_per_source[src] = counts_per_source.get(src, 0) + 1

    sources = sorted(list(counts_per_source.keys()))
    total_documents = len(sources)

    return {
        "total_chunks": total_chunks,
        "total_documents": total_documents,
        "sources": sources,
        "counts_per_source": counts_per_source,
    }


def clear_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """
    Explicitly reset or delete the vector store collection.
    Only executed when explicitly called to prevent accidental data loss.

    Args:
        persist_directory: Directory where Chroma stores files.
        collection_name: Collection to delete/reset.

    Returns:
        True if successfully cleared.
    """
    target_dir = Path(persist_directory or DEFAULT_PERSIST_DIRECTORY).resolve()
    logger.warning(f"Explicit clear requested for vector store at '{target_dir}'.")

    if target_dir.exists():
        try:
            embeddings = get_embedding_model()
            store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(target_dir),
            )
            store.delete_collection()
        except Exception as e:
            logger.debug(f"Collection deletion notice: {e}")

        for item in target_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink(missing_ok=True)
                except Exception:
                    pass

    return True
