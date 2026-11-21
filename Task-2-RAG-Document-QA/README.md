# Document Q&A Assistant with RAG & Vector Embeddings

> **Task 2: RAG Pipeline & Document Q&A**  
> *Datheon Python Generative AI & LangChain Internship*

An enterprise-grade, production-ready Retrieval-Augmented Generation (RAG) assistant that ingests, parses, chunks, and indexes unstructured documents (PDF and TXT) into a persistent ChromaDB vector store, answering questions with factual grounding and verifiable source citations using Google Gemini.

---

## 1. Project Overview

Unstructured documents (such as enterprise policies, product manuals, invoices, and contracts) contain vital information that is difficult to search efficiently with traditional keyword search. This project provides a comprehensive, end-to-end question-answering pipeline using modern LangChain 0.3.x, local embeddings (`sentence-transformers`), ChromaDB vector store, and Google Gemini (`gemini-2.5-flash`).

### Core Value Proposition
- **Factual Grounding:** Strict system prompts instruct the model to answer *only* based on retrieved document excerpts, completely mitigating hallucinations.
- **Verifiable Citations:** Every answer cites the source document filename and page number.
- **Local Embeddings:** Embeddings run locally on CPU (`sentence-transformers/all-MiniLM-L6-v2`), avoiding cloud embedding API costs or rate limits.
- **Dual Interface:** Full-featured interactive Streamlit web dashboard and headless CLI for batch ingestion and querying.

---

## 2. Key Features

- **Multi-Document Ingestion:** Native support for `.pdf` (using `PyPDFLoader`) and `.txt` files (bounded to a strict **20 MB per file limit**) with automatic metadata standardization.
- **Deterministic Chunking:** Parameter-validated recursive text splitting preserving source filename and page numbers across all chunks.
- **Persistent Vector Database:** ChromaDB storage maintaining indexed document chunks between application restarts.
- **Top-K Similarity Retrieval:** Configurable similarity search with defensive empty-database handling and error resilience.
- **Chunk Explorer:** Dedicated interactive inspector tab to browse, search, filter, and inspect raw stored chunks, character counts, chunk IDs, and page numbers directly from ChromaDB without regenerating embeddings.
- **Retrieval Inspector:** Expandable inspection panel under every answer displaying the exact chunks evaluated by Gemini, their retrieval rank, source filename, page, and similarity distance score.
- **Conversational Memory:** Streamlit session state retaining complete chat history while keeping chat clearing strictly decoupled from vector database resets.
- **Safe Administrative Controls:** Explicit confirmation gates for destructive vector database operations.
- **100% Mocked Automated Tests:** Fast, offline unit test suite requiring zero paid API credits or network downloads.

---

## 3. Architecture

```mermaid
flowchart TD
    A[Unstructured Document\nPDF / TXT max 20MB] --> B[document_loader.py\nPyPDFLoader & TextLoader]
    B --> C[chunking.py\nRecursiveCharacterTextSplitter\nsize=1000, overlap=200]
    C --> D[embeddings.py\nsentence-transformers\nall-MiniLM-L6-v2]
    D --> E[(ChromaDB\nchroma_db/ Collection)]
    
    F[User Question] --> G[retriever.py\nSimilarity Search Top-K=4]
    E -.->|Nearest Neighbor Chunks| G
    G --> H[qa_chain.py\nChatPromptTemplate + Context Delimiters]
    H --> I[Google Gemini\ngemini-3.1-flash-lite\ntemp=0.0]
    I --> J[Grounded Answer with\n[Source: file, Page: N] Citations]
    J --> K[Streamlit UI / CLI Output]
```

---

## 4. Technology Stack

- **Language:** Python 3.11.5 (Windows 11 verified)
- **Framework:** LangChain 0.3.x (`langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-chroma`, `langchain-huggingface`, `langchain-google-genai`)
- **LLM:** Google Gemini (`gemini-2.5-flash` / `ChatGoogleGenerativeAI`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB (`chromadb>=0.5.23`)
- **PDF Extraction:** `pypdf`
- **Web UI:** Streamlit (`streamlit>=1.41.1`)
- **Configuration:** `python-dotenv`
- **Testing:** `pytest` with `unittest.mock`

---

## 5. Folder Structure

```
Task-2-RAG-Document-QA/
├── app/
│   ├── __init__.py           # Package marker
│   ├── main.py               # CLI entry point
│   ├── document_loader.py    # PDF and TXT loaders with metadata standardization
│   ├── chunking.py           # Recursive text splitting & parameter validation
│   ├── embeddings.py         # Local HuggingFace embeddings with lazy caching
│   ├── vector_store.py       # ChromaDB persistence, similarity search & reset
│   ├── retriever.py          # Top-k similarity retrieval layer
│   └── qa_chain.py           # Grounded Gemini QA chain with citation formatting
├── data/                     # Uploaded documents (auto-persisted, gitignored)
├── tests/                    # 37 comprehensive unit & integration tests
│   ├── test_document_loader.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_qa_chain.py
│   └── test_streamlit_app.py
├── chroma_db/                # Persistent vector database directory (gitignored)
├── streamlit_app.py          # Streamlit interactive web dashboard
├── requirements.txt          # Windows & Python 3.11 compatible dependency pins
├── .env                      # Local secret keys (gitignored)
├── .env.example              # Template environment file
├── .gitignore                # Git exclusions
└── README.md                 # Complete project documentation
```

---

## 6. Python Environment Setup

```powershell
# Navigate into Task 2 directory
cd Task-2-RAG-Document-QA

# Create a virtual environment using Python 3.11
python -m venv .venv

# Activate the virtual environment on Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

---

## 7. Dependency Installation

```powershell
# Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install all verified dependencies
python -m pip install -r requirements.txt
```

---

## 8. API Key Configuration

1. Copy the example environment file:
   ```powershell
   cp .env.example .env
   ```
2. Obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3. Set your API key in `.env`:
   ```env
   GOOGLE_API_KEY=AIzaSy...your_actual_key_here
   GEMINI_API_KEY=AIzaSy...your_actual_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

---

## 9. Running the Streamlit Application

Launch the full-featured web dashboard in your browser:

```powershell
streamlit run streamlit_app.py
```

### Dashboard Capabilities:
- **Sidebar:** Upload multiple PDF/TXT files, inspect file sizes, click **"Process Documents"** to chunk and embed into ChromaDB, and view indexed document metrics.
- **Chat Interface:** Ask questions in real-time, view grounded answers with citations, clear conversation history, or perform confirmed vector database wipes.

---

## 10. Running the Command-Line Interface (CLI)

The CLI supports headless batch ingestion, querying, and database maintenance:

### A. Ingest a Single Document
```powershell
python -m app.main --file data/sample.txt
```

### B. Ingest an Entire Directory of Documents
```powershell
python -m app.main --dir data/
```

### C. Ask a Question with Citations
```powershell
python -m app.main --query "What is the policy renewal deadline?" --k 4
```

### D. Clear the Vector Database
```powershell
python -m app.main --clear
```

---

## 11. Running Automated Tests

Run the full automated test suite completely offline (no live API calls or model downloads required):

```powershell
pytest -q
```

Expected output:
```text
.....................................                                    [100%]
============================== 37 passed in 6.15s ==============================
```

---

## 12. Example Usage & Interaction

### Step 1: Upload and Index Documents
Upload a document like `company_policy.pdf`. The system extracts text across pages, segments into 1000-character chunks with 200-character overlap, generates 384-dimensional dense vectors, and commits to ChromaDB.

### Step 2: Querying the Assistant
- **User Question:** *"What is the standard warranty duration and return policy?"*
- **Assistant Response:**
  ```text
  The standard warranty coverage duration is 24 months from the original date of purchase. Customers are eligible for a full refund within 30 days provided the item is in its original packaging.

  📍 Sources:
  - company_policy.pdf (Page 4)
  - company_policy.pdf (Page 5)
  ```

### Step 3: Out-of-Domain / Insufficient Context Handling
- **User Question:** *"What was the company's Q3 revenue in 2021?"*
- **Assistant Response:**
  ```text
  I cannot find the answer to this question in the provided documents.
  ```
