# Multi-Tool Autonomous AI Agent (Task 3)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-green.svg)](https://python.langchain.com/)
[![Tests](https://img.shields.io/badge/pytest-49%20passed-success.svg)](https://pytest.org/)

An autonomous AI research and computation agent built for **Datheon Python Internship — Task 3**. The agent dynamically decomposes compound, multi-step queries, decides which tools to invoke sequentially, passes intermediate observations between steps, and formulates verifiable final answers without exposing private reasoning tokens.

---

## 1. Project Overview

Modern AI assistants must do more than recite static knowledge—they must autonomously interact with their environment, search the live web for ground-truth statistics, compute complex arithmetic without hallucination, and securely parse local enterprise files.

This project delivers:
- **Autonomous Tool-Calling Loop**: Uses LangChain to orchestrate tool invocations.
- **Real-Time Web Search**: Queries DuckDuckGo (`ddgs`) or Tavily, with automatic conversational query cleaning, source authority classification (.gov, .edu, multilateral agencies), and ranking.
- **Safe Mathematical Calculator**: Safe AST-based arithmetic parser that prevents unrestricted `eval()` or arbitrary code execution.
- **Secure Local File Reader**: Reads `.txt` and `.pdf` files strictly bounded to a designated `data/` directory with path-traversal prevention, true character counting, and scanned/image-only OCR notices.
- **Interactive Console & Single-Query CLI**: Provides observable action traces (`[Action]`, `[Tool]`, `[Input]`, `[Observation]`, `[Final Answer]`).
- **100% Offline Test Suite**: 49 unit and integration tests using deterministic mocks.

---

## 2. Architecture & Folder Structure

```
Task-3-Multi-Tool-AI-Agent/
├── .env.example              # Environment variables template
├── .env                      # Local environment secrets (Git-ignored)
├── .gitignore                # Git ignore rules
├── requirements.txt          # Pinned production dependencies
├── README.md                 # Project documentation
├── data/                     # Designated safe directory for local documents
│   ├── sample.txt            # Company financial summary sample
│   ├── company_policy.txt    # Enterprise HR policy sample
│   └── annual_report.pdf     # Sample PDF document
├── app/
│   ├── __init__.py
│   ├── main.py               # Interactive Console & CLI entrypoint
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Centralized configuration dataclass
│   │   └── logger.py         # Formatted structured logging
│   ├── tools/
│   │   ├── __init__.py       # Tool registry exports
│   │   ├── calculator.py     # AST-based safe math evaluator
│   │   ├── file_reader.py    # Sandboxed TXT & PDF extractor
│   │   └── web_search.py     # DuckDuckGo & Tavily search provider
│   └── agent/
│       ├── __init__.py
│       ├── llm.py            # Gemini / Gemma model factory
│       ├── callbacks.py      # Observable trace callback handler
│       └── executor.py       # Multi-tool agent execution engine
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures & mock chat models
    ├── test_calculator.py    # 16 unit tests for math operations & security
    ├── test_file_reader.py   # 11 unit tests for TXT/PDF loading & path traversal
    ├── test_web_search.py    # 6 unit tests for search parsing & error handling
    ├── test_agent_executor.py# 5 integration tests for multi-step execution & limits
    └── test_cli.py           # 4 unit tests for command-line interface & flags
```

### Architectural Execution Flow

```mermaid
flowchart TD
    User([User Prompt]) --> CLI[Interactive Console / CLI]
    CLI --> Agent[LangChain Tool-Calling Agent]
    
    subgraph Agent Loop
        Agent --> Decision{Agent Decision}
        Decision -->|Search Web| WS[web_search: DuckDuckGo / Tavily]
        Decision -->|Evaluate Math| CALC[calculator: Safe AST Evaluator]
        Decision -->|Read Local Data| FR[file_reader: Sandboxed TXT & PDF]
        
        WS --> Obs[Observation Feedback]
        CALC --> Obs
        FR --> Obs
        Obs --> Trace[Observable Agent Callback\n[Action] -> [Tool] -> [Observation]]
        Trace --> Decision
    end
    
    Decision -->|Sufficient Info Gathered| Final[Grounded Final Answer]
    Final --> CLI
```

---

## 3. Tool Specifications

### A. Web Search (`web_search`)
- **Default Engine**: DuckDuckGo Search via the maintained `ddgs` package (with fallback support).
- **Optional Engine**: Tavily Search (`tavily-python`) if `SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` are configured.
- **Source Authority & Provenance**: Automatically classifies source URLs into credibility tiers (High Authority: `.gov`, `.edu`, `.ac.uk`, multilateral agencies like UN/World Bank; Standard Web; Low Authority: user forums like Quora/Reddit). Prioritizes authoritative sources at the top of results.
- **Query Sanitation**: Automatically strips conversational filler (`"search for"`, `"please find"`) before issuing queries to maximize search engine recall.
- **Output**: Structured markdown entries containing `Title`, `URL`, `Credibility`, and `Snippet`. Never fabricates search results.

### B. Mathematical Calculator (`calculator`)
- **Engine**: Custom Python `ast` (Abstract Syntax Tree) recursive evaluator.
- **Zero Insecure `eval()`**: Strict node whitelist (`Add`, `Sub`, `Mult`, `Div`, `FloorDiv`, `Mod`, `Pow`, `UnaryOp`).
- **Features**:
  - Precedence handling: `(2 + 3) * 4` = `20`.
  - Power operator: both `**` and `^` (e.g. `2^8` = `256`).
  - Percentages: parses phrases like `10% of 500` or `15% * 200`.
  - Safe functions & constants: `sqrt()`, `abs()`, `round()`, `min()`, `max()`, `pi`, `e`.
- **Defensive Safeguards**: Zero division detection, integer overflow protection (powers > 1000 blocked), and max node limit (100 nodes).

### C. Local File Reader (`file_reader`)
- **Formats**: `.txt` (UTF-8 encoded) and `.pdf` (stream-extracted via `pypdf`).
- **True Content Verification**: Accurately counts extracted characters across pages. If a document contains zero extractable text, returns an informative notice (distinguishing blank documents from scanned image documents requiring OCR).
- **Sandbox Security**: Enforces path resolution against `config.data_dir`. Any relative directory escape attempt (`../../etc/passwd`) or absolute external path is blocked with a `Security Error: Access denied`.
- **Validation**: Rejects directories, empty files (0 bytes), oversized files (>10 MB), and unsupported formats (`.docx`, `.csv`).

---

## 4. Setup & Installation

### Prerequisites
- Python 3.11.5 (Recommended)
- Windows 11 / Linux / macOS
- Google Gemini API key (Free tier available at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Step 1: Navigate to Task 3
```powershell
cd "c:\Users\Divakar K R\Datheon-Python-Internship\Task-3-Multi-Tool-AI-Agent"
```

### Step 2: Activate the Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Update `.env` with your Google Gemini API key:
```env
GEMINI_API_KEY="your-google-gemini-api-key"
GEMINI_MODEL=gemma-4-26b-a4b-it
SEARCH_PROVIDER=duckduckgo
MAX_ITERATIONS=10
DATA_DIR=data
```

---

## 5. How to Run the Agent

### Interactive Console Session
Start the interactive multi-turn agent:
```powershell
python app/main.py
```

#### Available In-Session Commands
- `tools` — View all registered agent tools and descriptions.
- `files` — List all readable files in the local `data/` directory.
- `clear` — Clear the terminal screen and display the welcome banner.
- `exit` / `quit` — Exit the session.

### Single-Query Non-Interactive Mode
Execute a compound query directly from the command line:
```powershell
# Research and calculation query
python app/main.py --query "Search for India's population and calculate 5% of it."

# Local file inspection and calculation query
python app/main.py --query "Read sample.txt to find Datheon's annual revenue and calculate 10% of it."
```

### Inspection Flags
```powershell
# Display registered tools table
python app/main.py --tools

# Display readable data files table
python app/main.py --files
```

---

## 6. Example Observable Execution Trace

```
User Request: Search for India's population and calculate 5% of it.

[Action] Selecting web_search.
[Tool] web_search
[Input] {'query': 'current population of India 2024 Worldometer'}
[Observation] [1] India Demographics 2024 (Population, Age, Sex, Trends) - Worldometer
    URL: https://srv1.worldometers.info/demographics/india-demographics/
    Snippet: Population of India: 1,450,935,791 ...
[Action] Analyzing observation to determine next step.

[Action] Selecting calculator.
[Tool] calculator
[Input] {'expression': '1450935791 * 0.05'}
[Observation] Result: 72546789.55
[Action] Analyzing observation to determine next step.

[Final Answer]
Based on the data retrieved from Worldometer, the population of India in 2024 was approximately 1,450,935,791.

5% of India's population is 72,546,789.55.
```

---

## 7. Automated Testing

All tests run **100% offline** without requiring active internet connectivity, API credits, or external weight downloads:
```powershell
pytest -q
```

### Test Coverage Summary (49 Tests Passing)
| Test Suite | Tests | Scope |
| :--- | :---: | :--- |
| `tests/test_calculator.py` | 16 | Arithmetic, operator precedence, carets, powers, percentages, math functions, division by zero, code injection blocking, exponent overflow. |
| `tests/test_file_reader.py` | 12 | TXT parsing, text PDF page extraction, blank PDF notice, scanned image notice, empty files, missing files, directory targets, unsupported extensions, path traversal security. |
| `tests/test_web_search.py` | 12 | Query cleaning, source authority tiers (.gov, .edu, un.org vs quora/reddit), authority-based result sorting, DuckDuckGo mocked results, empty results, network timeouts, Tavily fallback, empty query errors, LangChain tool contract. |
| `tests/test_agent_executor.py` | 5 | Tool registration, single tool dispatch, multi-step sequential execution, max iteration limits, callback step logging. |
| `tests/test_cli.py` | 4 | `--tools`, `--files`, `--query` single run, missing API key handling. |

---

## 8. Quality Verification Checks

```powershell
# 1. Verify dependency tree consistency
python -m pip check

# 2. Run Python bytecode compilation
python -m compileall app tests

# 3. Run automated test suite
pytest -q
```

---

## 9. Security & Safety Guarantees

1. **Zero `eval()` Vulnerabilities**: The calculator parses raw expressions into an Abstract Syntax Tree (AST), rejecting non-whitelisted nodes and preventing code injection.
2. **Directory Sandboxing**: The file reader strictly enforces `is_relative_to(config.data_dir)` checks. Relative directory escapes (`../../`) and absolute paths outside `data/` are rejected with security errors.
3. **Execution Iteration Safeguard**: Configurable `MAX_ITERATIONS` prevents infinite agent loops or runaway API usage.
4. **Credential Isolation**: API keys are read via `python-dotenv` from local `.env` files that are strictly excluded by `.gitignore`.
