# Tech Stack

This document outlines the technologies, libraries, frameworks, and infrastructure used in the implementation of the **Hierarchical Semantic Routing for Scalable Agentic Retrieval in Federated Enterprise Environments** project.

---

# Core Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.12+** | Primary programming language used for backend development |
| **scikit-learn** | Implements `KMeans` and `MiniBatchKMeans` clustering algorithms, silhouette scoring for automatic cluster optimization |
| **sentence-transformers** | Uses the `all-MiniLM-L6-v2` model to convert queries into 384-dimensional semantic embeddings |
| **NumPy** | Efficient numerical computation, embedding operations, vector normalization, and similarity calculations |

---

# Agent Framework

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Orchestrates the complete multi-agent workflow using a graph-based state machine (`StateGraph`) |
| **LangChain Core** | Provides `HumanMessage`, `AIMessage`, and `BaseMessage` abstractions for inter-agent communication |
| **LangChain Ollama** | Executes local Large Language Models (`llama3.2`) using `ChatOllama` |
| **LangChain OpenAI** | Cloud-based LLM fallback for complex supervisor routing and classification tasks |

---

# Clustering Module

**Location**

```
backend/agents/clustering/
```

| File | Technologies Used |
|------|-------------------|
| `embedding_cluster.py` | `sentence-transformers`, `KMeans`, `MiniBatchKMeans`, `silhouette_score`, `pickle` for model persistence |

**Responsibilities**

- Semantic query embedding generation
- Automatic intent cluster discovery
- Cluster model persistence
- Intent prediction for incoming queries

---

# MCP Routing Module

**Location**

```
backend/agents/mcp_routing/
```

| File | Technologies Used |
|------|-------------------|
| `agent_tools.py` | Custom `AgentTool` dataclass and `AgentToolRegistry` for self-contained tool definitions |
| `tool_router.py` | `sentence-transformers`, `KMeans`, cosine similarity-based semantic tool routing |

**Responsibilities**

- Tool registration
- Tool clustering
- Semantic tool selection
- MCP-compatible routing

---

# Backend Infrastructure

| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API framework serving the LangGraph agent pipeline |
| **ChromaDB** | Vector database for semantic document retrieval |
| **SQLAlchemy** | ORM for managing users, sessions, and document metadata |

---

# Machine Learning Components

- Sentence Embedding Model
  - `all-MiniLM-L6-v2`

- Clustering Algorithm
  - `KMeans`
  - `MiniBatchKMeans`

- Similarity Metric
  - Cosine Similarity

- Cluster Optimization
  - Silhouette Score

---

# Project Architecture Stack

```
Frontend
    │
    ▼
FastAPI REST API
    │
    ▼
LangGraph Supervisor
    │
    ├── Keyword Routing
    ├── MCP Tool Routing
    ├── K-Means Intent Routing
    └── LLM-Based Routing
             │
             ▼
      Specialized Agents
      ├── Retrieval Agent
      ├── Summary Agent
      ├── Comparison Agent
      ├── Citation Agent
      └── Verification Agent
             │
             ▼
        ChromaDB Vector Store
```

---

# Query Processing Pipeline

```text
User Query
      │
      ▼
Sentence Transformer
(384-dimensional embedding)
      │
      ▼
K-Means Intent Clustering
      │
      ▼
MCP Tool Router
(Tool Cluster Selection)
      │
      ▼
LangGraph Supervisor
      │
      ├── Keyword Routing
      ├── MCP Routing
      ├── K-Means Routing
      └── LLM Fallback
      │
      ▼
Specialized Agent
      │
      ▼
Citation Generation
      │
      ▼
Self Verification
      │
      ▼
Final Response
```

---

# Technology Summary

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.12+ |
| Agent Framework | LangGraph, LangChain Core |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`) |
| Machine Learning | scikit-learn |
| Numerical Computing | NumPy |
| Local LLM | Ollama (`llama3.2`) |
| Cloud LLM | OpenAI |
| API Framework | FastAPI |
| Vector Database | ChromaDB |
| ORM | SQLAlchemy |
| Persistence | Pickle |
| Routing | K-Means + Cosine Similarity + MCP |
| Architecture | Hierarchical Semantic Multi-Agent Retrieval |