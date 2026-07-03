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
    
    # Query ChromaDB (retrieve top 6 chunks for dense context)
    retrieved = chroma_service.query(
        user_id=user_id,
        query_text=query,
        document_ids=selected_docs,
        top_k=6
    )
    
    return {
        "retrieved_context": retrieved,
        "next_agent": "citation_agent"
    }
