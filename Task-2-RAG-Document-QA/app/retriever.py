"""
Document retriever layer with configurable k, robust error handling,
and source metadata verification.
"""

from typing import List, Optional, Tuple
import logging
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.vector_store import get_vector_store, similarity_search

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_K = 4


def get_retriever(
    vector_store: Optional[Chroma] = None,
    k: int = DEFAULT_RETRIEVAL_K,
    search_type: str = "similarity",
):
    """
    Configure and return a LangChain VectorStoreRetriever.

    Args:
        vector_store: Optional Chroma instance. If None, initializes default.
        k: Number of relevant chunks to retrieve (default: 4).
        search_type: Search strategy ('similarity' or 'mmr').

    Returns:
        VectorStoreRetriever instance.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    store = vector_store or get_vector_store()
    return store.as_retriever(search_type=search_type, search_kwargs={"k": k})


def retrieve_relevant_documents(
    query: str,
    vector_store: Optional[Chroma] = None,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve top-k documents matching the query with defensive error handling.

    Args:
        query: Question or search string.
        vector_store: Chroma vector store instance.
        k: Number of chunks to retrieve (default: 4).

    Returns:
        List of matching Document chunks.
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to retriever.")
        return []

    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    store = vector_store or get_vector_store()

    try:
        if hasattr(store, "_collection") and store._collection is not None:
            count = store._collection.count()
            if count == 0:
                logger.info("Chroma collection is empty; returning empty results.")
                return []
    except Exception as exc:
        logger.debug(f"Unable to read collection count: {exc}")

    try:
        results = similarity_search(vector_store=store, query=query.strip(), k=k)
        return results
    except Exception as exc:
        logger.error(f"Error during retrieval: {exc}", exc_info=True)
        return []


def retrieve_relevant_documents_with_scores(
    query: str,
    vector_store: Optional[Chroma] = None,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k documents matching the query along with similarity/distance scores.

    Args:
        query: Question or search string.
        vector_store: Chroma vector store instance.
        k: Number of chunks to retrieve (default: 4).

    Returns:
        List of (Document, score) tuples.
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to retriever.")
        return []

    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    store = vector_store or get_vector_store()

    try:
        if hasattr(store, "_collection") and store._collection is not None:
            count = store._collection.count()
            if count == 0:
                logger.info("Chroma collection is empty; returning empty results.")
                return []
    except Exception as exc:
        logger.debug(f"Unable to read collection count: {exc}")

    try:
        # LangChain Chroma similarity_search_with_score returns List[Tuple[Document, float]]
        results = store.similarity_search_with_score(query=query.strip(), k=k)
        return results
    except Exception as exc:
        logger.error(f"Error during retrieval with scores: {exc}", exc_info=True)
        try:
            fallback_docs = similarity_search(vector_store=store, query=query.strip(), k=k)
            return [(d, 0.0) for d in fallback_docs]
        except Exception:
            return []
