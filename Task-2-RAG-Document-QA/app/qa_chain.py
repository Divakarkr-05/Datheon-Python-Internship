"""
Question-Answering RAG Chain powered by Google Gemini and LangChain.
Enforces grounded answers, citation accuracy (source file & page),
and defensive handling of missing API keys or insufficient context.
"""

from typing import Dict, Any, List, Optional
import os
import logging
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from app.retriever import retrieve_relevant_documents, retrieve_relevant_documents_with_scores

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

SYSTEM_PROMPT = """You are a precise, truthful, and helpful Document Question-Answering Assistant.

Your task is to answer the user's question STRICTLY based on the provided document excerpts below.

CRITICAL RULES:
1. ONLY use information explicitly stated in the context excerpts.
2. If the context does not contain sufficient factual information to answer the question, clearly state:
   "I cannot find the answer to this question in the provided documents."
   Do NOT attempt to guess, assume, or extrapolate outside the context.
3. When providing an answer, ALWAYS cite the source document name and page number from which the information was retrieved (e.g., "[Source: filename.pdf, Page: 1]").
4. Never invent, hallucinate, or fabricate source filenames or page citations.
5. Keep your answer concise, well-structured, and factual.

<CONTEXT>
{context}
</CONTEXT>
"""

USER_PROMPT = "{question}"


def get_google_api_key() -> str:
    """
    Resolve Google Gemini API key from environment variables.
    Checks GOOGLE_API_KEY first, then GEMINI_API_KEY.

    Raises:
        ValueError: If no API key is configured.
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Google API key not found. Please set 'GOOGLE_API_KEY' or 'GEMINI_API_KEY' "
            "in your environment or .env file."
        )
    return key.strip()


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    transport: str = "rest",
) -> ChatGoogleGenerativeAI:
    """
    Initialize and return a ChatGoogleGenerativeAI instance.

    Args:
        model_name: Target Gemini model (default: gemini-3.1-flash-lite or env var).
        temperature: Sampling temperature (default: 0.0 for deterministic answers).
        api_key: Optional API key override.
        transport: Communication transport ("rest" recommended for Windows compatibility).

    Returns:
        ChatGoogleGenerativeAI model instance.
    """
    resolved_key = api_key or get_google_api_key()
    resolved_model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=resolved_key,
        temperature=temperature,
        transport=transport,
    )


def format_context_chunks(documents: List[Document]) -> str:
    """
    Format retrieved document chunks into structured context with metadata headers.

    Args:
        documents: List of retrieved Document instances.

    Returns:
        Formatted context string.
    """
    if not documents:
        return "No relevant document excerpts found."

    formatted_parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        header = f"--- Excerpt {i} [Source: {source}, Page: {page}] ---"
        formatted_parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_parts)


def build_qa_chain(llm: Optional[ChatGoogleGenerativeAI] = None):
    """
    Build a standard LangChain runnable pipeline: Prompt -> LLM -> OutputParser.

    Args:
        llm: ChatGoogleGenerativeAI instance (initialized if None).

    Returns:
        Runnable chain.
    """
    active_llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    return prompt | active_llm | StrOutputParser()


def answer_question(
    query: str,
    vector_store=None,
    llm: Optional[ChatGoogleGenerativeAI] = None,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Full end-to-end RAG question answering pipeline.

    Args:
        query: User question.
        vector_store: Chroma vector store instance.
        llm: Optional ChatGoogleGenerativeAI instance.
        k: Number of context chunks to retrieve.

    Returns:
        Dictionary containing:
        - "query": Input question
        - "answer": LLM generated answer
        - "source_documents": List of retrieved Document instances
        - "citations": List of formatted citation strings
    """
    if not query or not query.strip():
        return {
            "query": query,
            "answer": "Please provide a valid question.",
            "source_documents": [],
            "citations": [],
            "retrieved_context": [],
        }

    # 1. Retrieve relevant chunks with distance/similarity scores
    scored_docs = retrieve_relevant_documents_with_scores(query=query, vector_store=vector_store, k=k)
    docs = [doc for doc, _ in scored_docs]

    if not docs:
        return {
            "query": query,
            "answer": "I cannot find the answer to this question in the provided documents.",
            "source_documents": [],
            "citations": [],
            "retrieved_context": [],
        }

    # 2. Extract citations
    citations = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        pg = doc.metadata.get("page", 1)
        citation_str = f"{src} (Page {pg})"
        if citation_str not in citations:
            citations.append(citation_str)

    # 3. Format detailed retrieved context for inspection
    retrieved_context = []
    for rank, (doc, score) in enumerate(scored_docs, start=1):
        retrieved_context.append({
            "rank": rank,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 1),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score": round(score, 4) if isinstance(score, (int, float)) else None,
        })

    # 4. Format context and invoke chain
    context_text = format_context_chunks(docs)
    chain = build_qa_chain(llm=llm)

    response_text = chain.invoke({
        "context": context_text,
        "question": query.strip(),
    })

    return {
        "query": query,
        "answer": response_text.strip(),
        "source_documents": docs,
        "citations": citations,
        "retrieved_context": retrieved_context,
    }
