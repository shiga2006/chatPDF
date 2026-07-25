# Enterprise Agentic AI Knowledge Assistant

The **Enterprise Agentic AI Knowledge Assistant** is a full-stack document intelligence app for uploaded PDFs. It combines a **FastAPI** backend, a **Streamlit** frontend, **LangGraph** routing, **ChromaDB** retrieval, and local or cloud LLMs to support semantic search, summaries, comparisons, citations, and document management.

---

## Architecture Diagram

The backend uses a LangGraph workflow with a memory rewrite stage, a routing/planning stage, specialized agents, citation generation, and verification before returning a final answer.

```mermaid
graph TD
   User([User Query]) --> MemoryNode[Memory Node]
   MemoryNode --> Router[Domain Router]
   Router --> Planner[Planner]
   Planner --> Supervisor{Supervisor}

   Supervisor -->|Retrieval| RetrievalAgent[Retrieval Agent]
   Supervisor -->|Summary| SummaryAgent[Summary Agent]
   Supervisor -->|Comparison| ComparisonAgent[Comparison Agent]

   RetrievalAgent --> CitationAgent[Citation Agent]
   SummaryAgent --> CitationAgent
   ComparisonAgent --> CitationAgent

   CitationAgent --> Verification[Verification Node]
   Verification -->|High confidence| FinalAnswer([Final Answer])
   Verification -->|Retry once| RetrievalAgent
```

---

## Folder Structure

```
chatPDF/
├── .streamlit/
│   └── config.toml               # Streamlit UI theme configuration
├── backend/
│   ├── api/
│   │   └── routes.py             # REST endpoints for auth, documents, chat, summary, and preview
│   ├── agents/
│   │   ├── graph.py              # LangGraph orchestration and routing
│   │   ├── llm_factory.py        # LLM provider selection
│   │   ├── state.py              # Shared graph state schema
│   │   ├── memory/agent.py       # Query rewriting from chat history
│   │   ├── retriever/agent.py    # ChromaDB retrieval
│   │   ├── summary/agent.py      # Document summarization
│   │   ├── comparison/agent.py   # Document comparison
│   │   └── citation/agent.py     # Final answer + citation assembly
│   │   ├── clustering/           # K-Means query intent clustering module
│   │   └── mcp_routing/          # MCP tool definitions and routing
│   ├── auth/
│   │   └── security.py           # JWT auth and password hashing
│   ├── database/
│   │   └── connection.py         # SQLAlchemy engine and session setup
│   ├── models/
│   │   └── db_models.py          # User, document, chat, and evaluation report tables
│   ├── rag/
│   │   └── pdf_processor.py      # PDF extraction and chunking
│   ├── schemas/
│   │   └── api_schemas.py        # Pydantic request/response models
│   ├── services/
│   │   └── summary_service.py    # Summary generation logic
│   ├── vectorstore/
│   │   └── chroma_service.py     # Persistent ChromaDB wrapper
│   └── main.py                   # FastAPI app entrypoint
├── evaluation/
│   ├── ragas_eval.py             # RAGAS evaluation CLI
│   ├── run_eval.py               # Interactive evaluation runner
│   ├── sample_eval_dataset.jsonl # 14 benchmark Q&A samples
│   └── reports/                  # Generated evaluation reports
├── frontend/
│   ├── app.py                    # Streamlit app entrypoint and navigation
│   ├── utils.py                  # API client helpers
│   └── views/
│       ├── dashboard.py          # Metrics and analytics
│       ├── documents.py          # Upload, preview, and document management
│       ├── chat.py               # Chat workspace
│       ├── evaluation.py         # RAGAS Evaluation Lab page
│       └── settings.py           # Connection/settings page
├── chromadb/                     # Persistent vector store data
├── uploads/                      # Uploaded PDFs by user ID
├── chatpdf.db                    # SQLite fallback database used in local development
├── .env                          # Local configuration settings
├── requirements.txt              # Project dependencies
└── README.md                     # Platform documentation
```

---

## Technology Stack

- **Backend Framework**: FastAPI
- **Frontend**: Streamlit
- **Agentic Engine**: LangGraph and LangChain
- **Vector Database**: ChromaDB persistent client
- **SQL Database**: MySQL in production, SQLite fallback for local development
- **Embedding Model**: sentence-transformers `all-MiniLM-L6-v2`
- **LLM Providers**: Ollama or OpenAI through a shared factory
- **PDF Extraction**: PyMuPDF
- **Security**: JWT authentication and bcrypt password hashing
- **Clustering**: scikit-learn KMeans / MiniBatchKMeans (7 query intent clusters: policy, factual, procedural, numeric, comparison, summarization, definitional)
- **MCP Tool Routing**: Semantic K-Means clustering over AgentTool definitions for intelligent agent capability routing
- **Evaluation**: RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall) with CLI + API + Streamlit frontend

---

## Installation & Setup

### Prerequisite 1: Database Setup
The app is configured for MySQL by default, but the backend falls back to the local `chatpdf.db` SQLite file when MySQL is unavailable.

If you want the MySQL path, make sure a MySQL server is running and the credentials in `.env` match your database user.

### Prerequisite 2: Ollama Setup
1. Download and install [Ollama](https://ollama.com).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull the required model:
   ```bash
   ollama pull llama3.2
   ```

If that model is not available locally, the backend now falls back to an extractive summary for document summaries.

### Step 3: Install Dependencies
Install the Python dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Environment Variables (`.env`)
Create a `.env` file in the root directory:
```ini
DATABASE_URL=mysql+pymysql://root:rootpassword@localhost:3306/chatpdf_db
JWT_SECRET=supersecretjwtkeyforagenticassistant123!@#
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2
CHROMA_DIR=chromadb
UPLOAD_DIR=uploads
```

---

## How to Run

### 1. Launch FastAPI Backend Server
Run uvicorn from the project root:
```bash
uvicorn --app-dir E:/debris/chatPDF backend.main:app --host 127.0.0.1 --port 8000 --reload
```
On startup, FastAPI connects to the configured SQL database, creates any missing tables, and serves the REST API on port 8000.

### 2. Launch Streamlit Frontend App
In a separate terminal tab, run:
```bash
streamlit run E:/debris/chatPDF/frontend/app.py
```
This opens the web browser at `http://localhost:8501`.

---

## How Agentic AI, LangGraph, and RAG Work

### 1. Retrieval-Augmented Generation (RAG)
When a PDF is uploaded, it is parsed and chunked, then indexed into persistent ChromaDB with metadata for document, page, chunk, and owner.
When a user asks a question, the retriever pulls the most relevant chunks and the citation layer formats the answer with source references.

### 2. LangGraph State Machine
The graph starts with a memory rewrite node, then a router and planner, and finally a supervisor that chooses the correct agent.
- **Memory Node**: Rewrites follow-up questions into standalone queries.
- **Domain Router**: Assigns a coarse domain such as HR, finance, IT, or product.
- **Planner**: Splits multi-part retrieval questions into subtasks.
- **Supervisor**: Chooses retrieval, summary, or comparison via a 4-tier routing pipeline:
  1. **Keyword overrides** — direct route for "compare" / "summarize" queries
  2. **MCP cluster routing** — semantic tool similarity matching via K-Means
  3. **K-Means intent routing** — detects comparison/summarization vs retrieval
  4. **LLM fallback** — lowest priority, used when confidence is low
- **Citation Node**: Converts the agent output into a final answer and citation list.
- **Verification Node**: Scores grounding quality and retries retrieval once when confidence is low.

---

## API Documentation

- **`POST /register`**: Registers a new user. Returns a JWT access token.
- **`POST /login`**: Logs in a user. Returns a JWT access token.
- **`GET /dashboard`**: Returns key statistics, recent uploads, and storage space metrics.
- **`POST /upload`**: Takes a PDF file, parses it, creates semantic vectors, generates 5 suggested questions, and saves it.
- **`GET /documents`**: Lists all documents uploaded by the authenticated user.
- **`PUT /documents/{id}`**: Renames an uploaded document.
- **`DELETE /documents/{id}`**: Deletes a document and clears its chunks from ChromaDB.
- **`POST /chat`**: Takes a user query and runs it through the LangGraph agent pipeline.
- **`GET /history`**: Returns chat history messages grouped by session.
- **`GET /summary/{id}`**: Generates an executive, short, or bulleted summary for the document, with an extractive fallback when the configured LLM is unavailable.
- **`GET /preview/{id}`**: Serves raw PDF bytes for frontend iframe rendering.

---

## K-Means Clustering Module

**Location**: `backend/agents/clustering/embedding_cluster.py`

The `EmbeddingCluster` class uses `sentence-transformers` + `sklearn.cluster.KMeans` to group user queries and document chunks into 7 semantic intent clusters:

| Cluster ID | Label | Example |
|-----------|-------|---------|
| 0 | `policy_search` | "What is the leave policy?" |
| 1 | `factual_qa` | "Who is the IT support contact?" |
| 2 | `procedural_howto` | "How do I apply for leave?" |
| 3 | `numeric_analytics` | "What is the budget limit?" |
| 4 | `comparison` | "Compare the benefits package" |
| 5 | `summarization` | "Summarize this document" |
| 6 | `definitional` | "What is a reimbursement?" |

**Key features**:
- `fit()` / `incremental_fit()` — train and update clusters incrementally
- `predict_with_confidence()` — returns cluster ID + confidence score
- `diversify_retrieval()` — MMR-style topic diversification for document chunks
- `_auto_determine_clusters()` — silhouette score for optimal k
- Persistence via pickle to `chromadb/cluster_model.pkl`

---

## MCP Tool Routing Module

**Location**: `backend/agents/mcp_routing/`

Defines 5 agent capabilities as `AgentTool` objects in a registry, then uses K-Means to cluster tools by capability similarity for intelligent routing.

**Registered Tools**:
| Tool | Category | Description |
|------|----------|-------------|
| `retrieve_semantic` | retrieval | Semantic vector search across documents |
| `summarize_comprehensive` | summary | Generate document summaries |
| `compare_documents` | comparison | Side-by-side comparison of documents |
| `cite_sources` | citation | Format citations with source metadata |
| `verify_answer` | verification | Score grounding quality of answers |

**Routing Pipeline**:
1. Tools are clustered by description via K-Means
2. User query is embedded and mapped to the nearest tool cluster centroid
3. Best tool within the matched cluster is selected based on confidence threshold
4. Falls back to keyword matching when confidence is low

---

## RAGAS Evaluation System

The project includes a **RAGAS (Retrieval-Augmented Generation Assessment)** evaluation system for benchmarking the quality of RAG outputs. It measures four key metrics:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | How factually consistent the answer is with the retrieved context |
| **Answer Relevancy** | How relevant the generated answer is to the question |
| **Context Precision** | How relevant the retrieved context chunks are to the question |
| **Context Recall** | Whether all necessary context for answering the question was retrieved |

### CLI Usage

```bash
# Basic evaluation with the sample dataset
python evaluation/ragas_eval.py

# Custom input file
python evaluation/ragas_eval.py --input evaluation/sample_eval_dataset.jsonl

# Export as Markdown or HTML report
python evaluation/ragas_eval.py --format markdown --verbose
python evaluation/ragas_eval.py --format html

# Use different judge/embedding models
python evaluation/ragas_eval.py --judge-model gpt-4o --embedding-model text-embedding-3-large

# Quick runner (with interactive mode)
python evaluation/run_eval.py
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evaluate` | Run RAGAS evaluation on a set of question-answer-context-ground_truth samples |
| `GET` | `/evaluate/reports` | List all evaluation reports for the current user |
| `GET` | `/evaluate/reports/{id}` | Retrieve a specific evaluation report with full details |
| `DELETE` | `/evaluate/reports/{id}` | Delete an evaluation report |

### Dataset Format

The evaluation dataset uses JSONL (one JSON object per line) with the following schema:

```json
{
  "question": "What is the leave carry-forward limit?",
  "answer": "Employees can carry forward up to 10 leave days.",
  "contexts": [
    "Section 4.2: Unused paid leave may be carried forward, capped at 10 days per year.",
    "Section 4.3: Carried-forward leave expires after March 31."
  ],
  "ground_truth": "Employees may carry forward a maximum of 10 unused leave days to the next year."
}
```

The sample dataset includes 14 benchmark questions across HR, IT, Finance, and Legal domains.

### Frontend Evaluation Lab

The **Evaluation Lab** page in the Streamlit frontend allows you to:

1. **Run evaluations** — paste JSONL, upload a file, or use the sample dataset
2. **View past reports** — browse, view detailed per-sample scores, and delete reports
3. **Configure models** — choose the judge LLM (e.g., `gpt-4o-mini`, `gpt-4o`) and embedding model

### Requirements

The following dependencies are already in `requirements.txt`:
- `ragas` — RAGAS evaluation framework
- `datasets` — HuggingFace Datasets for data loading
- `pandas` — Data manipulation for score aggregation

An **OpenAI API key** (`OPENAI_API_KEY`) must be set in the environment for the judge LLM and embeddings used during evaluation.
