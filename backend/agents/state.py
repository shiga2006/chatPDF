from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Core state parameters
    messages: List[BaseMessage]
    user_id: int
    session_id: int
    selected_document_ids: Optional[List[int]]
    normalized_query: str
    selected_domain: str
    selected_tools: List[str]
    subtasks: List[str]
    retrieval_attempts: int
    verification_score: float
    verification_notes: str
    needs_reretrieval: bool
    
    # Routing parameter for supervisor
    next_agent: str
    
    # Inter-agent exchange parameters
    query: str
    retrieved_context: List[Dict[str, Any]]
    summary_result: str
    comparison_result: str
    citations: List[Dict[str, Any]]
    final_answer: str

    # --- K-Means & MCP Clustering fields ---
    query_intent_cluster: int           # K-Means cluster ID for query intent
    query_intent_label: str             # Human-readable cluster label
    query_intent_confidence: float      # Confidence of cluster assignment
    mcp_routing_decision: str           # MCP tool routing decision (category)
    mcp_routing_confidence: float       # MCP routing confidence score
    cluster_diversified_context: List[Dict[str, Any]]  # Diversity-ranked chunks
