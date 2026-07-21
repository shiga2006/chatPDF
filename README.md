# Enterprise Agentic AI Knowledge Assistant !

The **Enterprise Agentic AI Knowledge Assistant** is a production-grade, full-stack AI platform designed to index, query, analyze, and compare internal corporate PDF documentation. Powered by a multi-agent state graph orchestrated with **LangGraph**, it enables semantic retrieval, document summarization, side-by-side policy comparisons, and automatic citation mapping, backed by **FastAPI** on the backend and **Streamlit** on the frontend.

---

## Architecture Diagram

The application uses a multi-agent system built on **LangGraph**. A **Supervisor Agent** evaluates user queries and routes them to specialized agents. The state transitions logically, ensuring that citations are generated and verified before returning a response.

```mermaid
graph TD
    User([User Query]) --> MemoryNode[Conversation Memory Agent]
    MemoryNode -->|Standalone Query| Supervisor{Supervisor Agent}
    
    Supervisor -->|Q&A / Fact Search| RetrievalAgent[Retrieval Agent]
    Supervisor -->|Summarization| SummaryAgent[Summary Agent]
    Supervisor -->|File Comparison| ComparisonAgent[Comparison Agent]
    
    RetrievalAgent --> CitationAgent[Citation Agent]
    SummaryAgent --> CitationAgent
    ComparisonAgent --> CitationAgent
    
    CitationAgent --> FinalAnswer([Final Response + Citations])
```

---

## Folder Structure

```
chatPdf/
├── .streamlit/
│   └── config.toml               # Streamlit UI theme configurations
├── backend/
│   ├── api/
│   │   └── routes.py             # FastAPI routes (Auth, Upload, Chat, Summary)
│   ├── auth/
│   │   └── security.py           # JWT generation, validation, and password hashing
│   ├── database/
│   │   └── connection.py         # SQLAlchemy connection pool and MySQL database creation
│   ├── models/
│   │   └── db_models.py          # SQLAlchemy models (User, Document, ChatSession, Chat)
│   ├── schemas/
│   │   └── api_schemas.py        # Pydantic validation schemas
│   ├── services/
│   │   └── summary_service.py    # Document text extraction and summarization logic
│   ├── agents/
│   │   ├── retriever/
│   │   │   └── agent.py          # Retrieval Agent Node
│   │   ├── summary/
│   │   │   └── agent.py          # Summary Agent Node
│   │   ├── comparison/
│   │   │   └── agent.py          # Comparison Agent Node
│   │   ├── citation/
│   │   │   └── agent.py          # RAG text generation and Citation Agent Node
│   │   ├── memory/
│   │   │   └── agent.py          # Context Query Rewriter Agent Node
│   │   ├── state.py              # LangGraph shared State definitions
│   │   ├── llm_factory.py        # LLM client builder (Ollama / OpenAI)
│   │   └── graph.py              # Master LangGraph compilation
│   ├── rag/
│   │   └── pdf_processor.py      # PDF text extractor and chunker (PyMuPDF)
│   ├── vectorstore/
│   │   └── chroma_service.py     # ChromaDB wrapper (SentenceTransformers)
│   └── config.py                 # Pydantic configuration loader
├── frontend/
│   ├── views/
│   │   ├── dashboard.py          # Streamlit metrics cards and analytics grid
│   │   ├── documents.py          # Streamlit document manager, uploaders, and previews
│   │   ├── chat.py               # Streamlit ChatGPT-like interface and citations expander
│   │   └── settings.py           # Streamlit connection test dashboard
│   ├── app.py                    # Streamlit entry point (auth, routing, navigation)
│   └── utils.py                  # API client handlers and session handlers
├── uploads/                      # Local PDF uploads folder (grouped by User ID)
├── chromadb/                     # Local vector database storage folder
├── .env                          # Configuration settings
├── requirements.txt              # Project dependencies
└── README.md                     # Platform documentation
```

---

## Technology Stack

- **Backend Framework**: FastAPI (Asynchronous endpoints)
- **Frontend View**: Streamlit (Python-driven premium UI)
- **Database (SQL)**: MySQL (using SQLAlchemy ORM and PyMySQL)
- **Vector Database**: ChromaDB (Persistent client mode)
- **Agentic Engine**: LangGraph & LangChain
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Local execution)
- **Local Large Language Model**: Ollama (`llama3.2` or `llama3.1`)
- **PDF Extraction**: PyMuPDF (fitz)
- **Security**: JWT Authentication + Bcrypt hashing

---

## Installation & Setup

### Prerequisite 1: MySQL Setup
Ensure you have a MySQL server running locally. Create or verify a root database connection.
Create a database named `chatpdf_db` or let the backend auto-create it using your configured credentials.

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

### Step 3: Clone & Install Dependencies
1. Place the project files inside your workspace.
2. Install the python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Environment Variables (`.env`)
Create a `.env` file in the root directory (based on `.env.example`):
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
Run uvicorn from the root directory:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
On startup, FastAPI will automatically connect to MySQL and create the required database tables.

### 2. Launch Streamlit Frontend App
In a separate terminal tab, run:
```bash
streamlit run frontend/app.py
```
This opens the web browser at `http://localhost:8501`.

---

## How Agentic AI, LangGraph, and RAG Work

### 1. Retrieval-Augmented Generation (RAG)
When a PDF is uploaded, it is parsed page-by-page. The text is chunked into 1000-character blocks with a 200-character overlap. Each chunk is embedded using the `all-MiniLM-L6-v2` model and saved to a persistent ChromaDB database.
When a user asks a question, the vector DB is searched to retrieve the top-K relevant chunks, which are then passed to the LLM as grounding context to answer the question, eliminating hallucinations.

### 2. LangGraph State Machine
Using LangGraph, we define nodes as python functions and edges as conditional transitions.
- **Query Rewriting (Memory Agent)**: Standardizes questions like *"tell me more"* based on context, translating them to search phrases.
- **Supervisor Router**: Classifies the query using LLM/rules to route to the correct agent node.
- **Specialized Agents**: Summarize, compare, or search.
- **Citation Agent**: Parses the generated answers to ensure every claim maps to an original document and page number.

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
- **`GET /summary/{id}`**: Generates an executive, short, or bulleted summary for the document.
- **`GET /preview/{id}`**: Serves raw PDF bytes for frontend iframe rendering.
