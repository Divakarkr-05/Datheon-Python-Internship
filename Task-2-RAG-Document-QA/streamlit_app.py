"""
Document Q&A Assistant - Streamlit Web Dashboard
Datheon Python Generative AI & LangChain Internship - Task 2

Interactive RAG application allowing users to upload PDF/TXT documents,
index them into persistent ChromaDB, and query them with grounded Google Gemini answers.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import os
import shutil
import logging
from dotenv import load_dotenv
import streamlit as st

# Load environment configuration
load_dotenv()

from app.document_loader import load_document, SUPPORTED_EXTENSIONS
from app.chunking import chunk_documents
from app.vector_store import (
    get_vector_store,
    add_documents_to_vector_store,
    clear_vector_store,
    get_stored_chunks,
    get_stored_chunks_summary,
    DEFAULT_PERSIST_DIRECTORY,
)
from app.qa_chain import answer_question, get_google_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base upload directory
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ==============================================================================
# Helper Functions (Modular & Unit-Testable)
# ==============================================================================

def validate_uploaded_file(file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file format and content size.

    Args:
        file: Streamlit UploadedFile or file-like object.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if file is None:
        return False, "No file provided."

    name = getattr(file, "name", "")
    suffix = Path(name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

    size = getattr(file, "size", 0)
    if size == 0:
        return False, f"File '{name}' is empty (0 bytes)."
    if size > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File '{name}' exceeds the maximum allowed limit of {MAX_FILE_SIZE_MB} MB ({size / (1024 * 1024):.1f} MB).",
        )

    return True, None


def filter_chunks(
    chunks: List[Dict[str, Any]],
    source_filter: Optional[str] = None,
    search_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter stored chunks by source document name and/or keyword search query.

    Args:
        chunks: List of chunk dictionaries.
        source_filter: Selected source document name or 'All Sources'.
        search_query: Optional case-insensitive text substring to match.

    Returns:
        Filtered list of chunk dictionaries.
    """
    filtered = list(chunks)
    if source_filter and source_filter not in ("All", "All Sources"):
        filtered = [c for c in filtered if c.get("source") == source_filter]

    if search_query and search_query.strip():
        q_lower = search_query.strip().lower()
        filtered = [c for c in filtered if q_lower in c.get("content", "").lower()]

    return filtered


def save_uploaded_file(uploaded_file, target_dir: Optional[Path] = None) -> Path:
    """
    Save an uploaded file safely to the data directory, preventing collisions.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        target_dir: Destination folder path (defaults to DATA_DIR).

    Returns:
        Path of the saved local file.
    """
    dest_dir = target_dir or DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(uploaded_file.name).name  # sanitize
    target_path = dest_dir / file_name

    # Write file bytes
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return target_path


def process_uploaded_documents(
    uploaded_files: List[Any],
    persist_directory: Optional[Path] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Process, chunk, and index uploaded documents into ChromaDB.

    Args:
        uploaded_files: List of Streamlit uploaded files.
        persist_directory: Target ChromaDB folder.
        chunk_size: Chunk character limit.
        chunk_overlap: Overlap count.

    Returns:
        Dictionary containing processing metrics and status.
    """
    if not uploaded_files:
        return {"status": "error", "message": "No files selected.", "doc_count": 0, "chunk_count": 0, "files": []}

    saved_paths: List[Path] = []
    processed_filenames: List[str] = []

    # 1. Validate & Save
    for f in uploaded_files:
        is_valid, err = validate_uploaded_file(f)
        if not is_valid:
            return {"status": "error", "message": err, "doc_count": 0, "chunk_count": 0, "files": []}
        saved_path = save_uploaded_file(f)
        saved_paths.append(saved_path)
        processed_filenames.append(f.name)

    # 2. Ingest documents
    all_raw_docs = []
    for path in saved_paths:
        try:
            docs = load_document(path)
            all_raw_docs.extend(docs)
        except Exception as e:
            logger.error(f"Failed to load '{path.name}': {e}")
            return {
                "status": "error",
                "message": f"Error loading '{path.name}': {e}",
                "doc_count": 0,
                "chunk_count": 0,
                "files": [],
            }

    if not all_raw_docs:
        return {
            "status": "warning",
            "message": "No extractable text found in the provided documents.",
            "doc_count": 0,
            "chunk_count": 0,
            "files": processed_filenames,
        }

    # 3. Chunk documents
    chunks = chunk_documents(all_raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 4. Store in ChromaDB
    vector_store = get_vector_store(persist_directory=persist_directory)
    add_documents_to_vector_store(vector_store, chunks)

    return {
        "status": "success",
        "message": f"Successfully indexed {len(processed_filenames)} file(s) into {len(chunks)} chunks.",
        "doc_count": len(processed_filenames),
        "chunk_count": len(chunks),
        "files": processed_filenames,
    }


def submit_question(
    query: str,
    vector_store=None,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Validate query and execute RAG question answering.

    Args:
        query: User question string.
        vector_store: Chroma vector store instance.
        k: Top-k chunks to retrieve.

    Returns:
        Response dictionary with answer and citations.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return {
            "query": "",
            "answer": "Please enter a valid question before asking.",
            "citations": [],
            "source_documents": [],
            "retrieved_context": [],
        }

    try:
        # Check API key first
        get_google_api_key()
    except ValueError as ve:
        return {
            "query": clean_query,
            "answer": f"API Key Configuration Error: {ve}",
            "citations": [],
            "source_documents": [],
            "retrieved_context": [],
        }

    return answer_question(query=clean_query, vector_store=vector_store, k=k)


def clear_chat_history() -> None:
    """Clear conversation history from session state without modifying database."""
    st.session_state["messages"] = []


def reset_vector_database(persist_directory: Optional[Path] = None) -> bool:
    """
    Clear ChromaDB collection and remove local data files.
    Chat history is intentionally untouched.
    """
    success = clear_vector_store(persist_directory=persist_directory)
    # Clear local uploaded data files (preserve .gitkeep)
    for p in DATA_DIR.iterdir():
        if p.name != ".gitkeep":
            if p.is_file():
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

    if "indexed_files" in st.session_state:
        st.session_state["indexed_files"] = []
    if "total_chunks" in st.session_state:
        st.session_state["total_chunks"] = 0

    return success


# ==============================================================================
# Streamlit UI Rendering
# ==============================================================================

def init_session_state():
    """Initialize Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "indexed_files" not in st.session_state:
        st.session_state["indexed_files"] = []
    if "total_chunks" not in st.session_state:
        st.session_state["total_chunks"] = 0


def render_sidebar():
    """Render document management and vector store administrative controls."""
    with st.sidebar:
        st.header("📂 Document Management")
        st.markdown("Upload documents to build your vector knowledge base.")

        # API Key Status
        has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        if has_key:
            st.success("🟢 Gemini API Key Detected", icon="✅")
        else:
            st.error("🔴 Gemini API Key Missing (Check `.env`)", icon="⚠️")

        st.markdown("---")

        # A. File Uploader
        uploaded_files = st.file_uploader(
            "Select Documents (.pdf, .txt)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload one or multiple PDF or plain text files (max 20 MB each).",
        )

        if uploaded_files:
            st.caption(f"Selected {len(uploaded_files)} file(s):")
            for f in uploaded_files:
                st.caption(f"• **{f.name}** ({f.size / 1024:.1f} KB)")

        # B. Document Processing Button
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one document first.")
            else:
                with st.spinner("Chunking text & generating local embeddings..."):
                    result = process_uploaded_documents(uploaded_files)

                    if result["status"] == "success":
                        st.success(result["message"])
                        # Update session state tracking
                        current_set = set(st.session_state["indexed_files"])
                        current_set.update(result["files"])
                        st.session_state["indexed_files"] = sorted(list(current_set))
                        st.session_state["total_chunks"] += result["chunk_count"]
                    elif result["status"] == "warning":
                        st.warning(result["message"])
                    else:
                        st.error(result["message"])

        st.markdown("---")

        # E. Document Status & Database Management
        st.subheader("📊 Indexed Knowledge Base")
        if st.session_state["indexed_files"]:
            st.write(f"**Total Indexed Files:** {len(st.session_state['indexed_files'])}")
            st.write(f"**Total Chunks in DB:** ~{st.session_state['total_chunks']}")
            with st.expander("View Indexed Documents"):
                for doc_name in st.session_state["indexed_files"]:
                    st.write(f"📄 {doc_name}")
        else:
            st.info("No documents indexed yet. Upload and process documents above.")

        st.markdown("---")
        st.subheader("⚠️ Database Controls")
        confirm_clear = st.checkbox("Confirm database deletion", value=False)
        if st.button("🗑️ Clear Vector Database", use_container_width=True):
            if not confirm_clear:
                st.warning("Please check 'Confirm database deletion' above to proceed.")
            else:
                reset_vector_database()
                st.success("Vector database cleared successfully!")
                st.rerun()


def render_chunk_explorer():
    """Render the Chunk Explorer interface for inspecting ChromaDB contents."""
    st.subheader("🔍 Chunk Explorer & Vector Store Inspector")
    st.caption("Inspect raw stored document chunks, metadata, and character distributions directly from ChromaDB.")

    # 1. Fetch chunks & summary without regenerating embeddings
    summary = get_stored_chunks_summary()
    total_docs = summary.get("total_documents", 0)
    total_chunks = summary.get("total_chunks", 0)
    sources = summary.get("sources", [])
    counts_per_source = summary.get("counts_per_source", {})

    # 2. Metric Overview Cards
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Total Indexed Documents", total_docs)
    with col_m2:
        st.metric("Total Stored Chunks", total_chunks)

    if total_chunks == 0:
        st.info("The vector database is currently empty. Upload and process documents from the sidebar to inspect chunks.")
        return

    # 3. Document Breakdown
    with st.expander("📊 Chunk Distribution per Document", expanded=False):
        for doc_name, count in counts_per_source.items():
            st.write(f"• **{doc_name}**: `{count}` chunk(s)")

    st.markdown("---")

    # 4. Filter & Search Controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        source_options = ["All Sources"] + sources
        selected_source = st.selectbox("Filter by Source Document:", source_options, key="explorer_src_filter")
    with col_f2:
        search_keyword = st.text_input("Search Chunk Content:", placeholder="Type keywords...", key="explorer_search_filter")
    with col_f3:
        limit_choice = st.selectbox("Display Limit:", [10, 25, 50, "All"], index=1, key="explorer_limit_filter")

    # 5. Fetch and Filter Chunks
    raw_chunks = get_stored_chunks()
    matched_chunks = filter_chunks(raw_chunks, source_filter=selected_source, search_query=search_keyword)

    st.caption(f"Showing **{len(matched_chunks)}** of **{total_chunks}** stored chunk(s)")

    if not matched_chunks:
        st.warning("No chunks matched your current filter criteria.")
        return

    # Pagination slice
    if limit_choice != "All" and isinstance(limit_choice, int):
        display_chunks = matched_chunks[:limit_choice]
    else:
        display_chunks = matched_chunks

    # 6. Group Chunks by Source Filename
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in display_chunks:
        grouped.setdefault(c["source"], []).append(c)

    for src_name, chunk_list in grouped.items():
        st.markdown(f"#### 📄 {src_name} ({len(chunk_list)} displayed)")
        for c in chunk_list:
            header_title = (
                f"Chunk #{c['chunk_index']} • Page {c['page']} • {c['char_count']} chars • ID: {c['id'][:8]}..."
            )
            with st.expander(header_title, expanded=False):
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.write(f"**Source:** `{c['source']}`")
                    st.write(f"**Page Number:** `{c['page']}`")
                with col_d2:
                    st.write(f"**Chunk Index:** `{c['chunk_index']}`")
                    st.write(f"**Char Count:** `{c['char_count']}`")
                with col_d3:
                    st.write(f"**Chunk ID:** `{c['id']}`")

                st.markdown("**Complete Chunk Text:**")
                st.text_area(
                    "Raw Content:",
                    value=c["content"],
                    height=140,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"chunk_view_{c['id']}",
                )


def render_main_content():
    """Render Q&A area, conversation history, user input field, and Chunk Explorer."""
    st.title("📚 Document Q&A Assistant")
    st.markdown(
        "Ask questions about your uploaded documents with verifiable source citations, "
        "or inspect raw ChromaDB embeddings chunks in the Chunk Explorer."
    )

    tab_qa, tab_explorer = st.tabs(["💬 Question & Answer", "🔍 Chunk Explorer"])

    # TAB 1: QUESTION & ANSWER
    with tab_qa:
        # Chat Action Bar
        col_clear, col_status = st.columns([1, 4])
        with col_clear:
            if st.button("🧹 Clear Chat History", use_container_width=True):
                clear_chat_history()
                st.rerun()
        with col_status:
            if not st.session_state["indexed_files"]:
                st.caption("ℹ️ *Knowledge base is empty. Upload documents in the sidebar to enable answers.*")

        st.markdown("---")

        # Conversation History Display
        chat_container = st.container()
        with chat_container:
            if not st.session_state["messages"]:
                st.info("👋 Welcome! Upload a document on the left and ask your first question below.")
            else:
                for idx, msg in enumerate(st.session_state["messages"]):
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.write(msg["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(msg["content"])

                            # Verifiable Citations
                            if msg.get("citations"):
                                with st.expander("📍 Verifiable Sources & Citations", expanded=False):
                                    for citation in msg["citations"]:
                                        st.markdown(f"- **{citation}**")

                            # Retrieved Context Inspector
                            if msg.get("retrieved_context"):
                                with st.expander(
                                    f"🔎 Retrieved Context ({len(msg['retrieved_context'])} Chunks Evaluated by Gemini)",
                                    expanded=False,
                                ):
                                    for c_info in msg["retrieved_context"]:
                                        rank = c_info.get("rank", 1)
                                        src = c_info.get("source", "Unknown")
                                        pg = c_info.get("page", 1)
                                        score = c_info.get("score")
                                        score_str = f" • Distance Score: `{score}`" if score is not None else ""
                                        st.markdown(f"**Rank #{rank}** • `📄 {src}` (Page {pg}){score_str}")
                                        st.text_area(
                                            f"Content [Rank #{rank}]:",
                                            value=c_info.get("content", ""),
                                            height=100,
                                            disabled=True,
                                            key=f"retrieved_{idx}_{rank}",
                                        )

        # Question Input Field & Ask Button
        st.markdown("### 💬 Ask a Question")
        with st.form("qa_form", clear_on_submit=True):
            query_input = st.text_input(
                "Enter your question:",
                placeholder="e.g. What is the total invoice amount? or What are the candidate's core skills?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("🔍 Ask Question", type="primary")

        if submitted:
            clean_q = (query_input or "").strip()
            if not clean_q:
                st.warning("Please enter a question before submitting.")
            else:
                # 1. Record User Message
                st.session_state["messages"].append({"role": "user", "content": clean_q})

                # 2. Execute RAG Pipeline with Spinner
                with st.spinner("Searching document embeddings & reasoning with Gemini..."):
                    response = submit_question(query=clean_q)

                # 3. Record Assistant Response with Retrieved Context
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response.get("citations", []),
                    "retrieved_context": response.get("retrieved_context", []),
                })

                # 4. Refresh to show updated message stream
                st.rerun()

    # TAB 2: CHUNK EXPLORER
    with tab_explorer:
        render_chunk_explorer()


def main():
    """Main application layout and execution."""
    st.set_page_config(
        page_title="Document Q&A Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
