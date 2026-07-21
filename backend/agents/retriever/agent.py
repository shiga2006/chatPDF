from backend.agents.state import AgentState
from backend.vectorstore.chroma_service import chroma_service

def retrieval_agent(state: AgentState) -> dict:
    """
    Retrieval Agent Node:
    Queries the vector database (ChromaDB) to retrieve top-k chunks 
    relevant to the user query.
    """
    user_id = state.get("user_id")
    query = state.get("query")
    selected_docs = state.get("selected_document_ids")
    subtasks = state.get("subtasks", []) or [query]

    # Query ChromaDB across each planned subtask to support multi-hop retrieval.
    retrieved = []
    seen = set()
    for subtask in subtasks[:4]:
        subtask_results = chroma_service.query(
            user_id=user_id,
            query_text=subtask,
            document_ids=selected_docs,
            top_k=4
        )
        for item in subtask_results:
            meta = item.get("metadata", {})
            dedupe_key = (
                meta.get("document_id"),
                meta.get("page"),
                meta.get("chunk_index")
            )
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                retrieved.append(item)

    if not retrieved:
        retrieved = chroma_service.query(
            user_id=user_id,
            query_text=query,
            document_ids=selected_docs,
            top_k=6
        )
    
    return {
        "retrieved_context": retrieved,
        "next_agent": "citation_agent",
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
    }
