"""
CLI entry point for Document Q&A Assistant with RAG.
Supports document ingestion, vector store queries, and administrative commands.
"""

import argparse
import sys
from pathlib import Path
import logging

from app.document_loader import load_document, load_documents_from_directory
from app.chunking import chunk_documents
from app.vector_store import get_vector_store, add_documents_to_vector_store, clear_vector_store
from app.qa_chain import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Document Q&A Assistant with RAG & Vector Embeddings"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single document (.pdf or .txt) to ingest.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to a directory containing documents to ingest.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Question to ask against the indexed documents.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve (default: 4).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector store database.",
    )

    args = parser.parse_args()

    # Handle clearing vector store
    if args.clear:
        clear_vector_store()
        print("Vector store collection successfully cleared.")
        if not args.file and not args.dir and not args.query:
            return

    vector_store = get_vector_store()

    # Handle single file ingestion
    if args.file:
        file_path = Path(args.file)
        print(f"Loading document: {file_path.name}...")
        docs = load_document(file_path)
        chunks = chunk_documents(docs)
        add_documents_to_vector_store(vector_store, chunks)
        print(f"Ingested {len(chunks)} chunks into vector store.")

    # Handle directory ingestion
    if args.dir:
        dir_path = Path(args.dir)
        print(f"Loading documents from directory: {dir_path}...")
        docs = load_documents_from_directory(dir_path)
        chunks = chunk_documents(docs)
        add_documents_to_vector_store(vector_store, chunks)
        print(f"Ingested {len(chunks)} chunks into vector store.")

    # Handle question answering
    if args.query:
        print(f"\nQuestion: {args.query}")
        result = answer_question(query=args.query, vector_store=vector_store, k=args.k)
        print("\nAnswer:")
        print(result["answer"])
        if result["citations"]:
            print("\nSources:")
            for citation in result["citations"]:
                print(f"  - {citation}")


if __name__ == "__main__":
    main()
