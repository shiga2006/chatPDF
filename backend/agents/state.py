from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Core state parameters
    messages: List[BaseMessage]
    user_id: int
    session_id: int
    selected_document_ids: Optional[List[int]]
    
    # Routing parameter for supervisor
    next_agent: str
    
    # Inter-agent exchange parameters
    query: str
    retrieved_context: List[Dict[str, Any]]
    summary_result: str
    comparison_result: str
    citations: List[Dict[str, Any]]
    final_answer: str
