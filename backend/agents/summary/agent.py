import logging
from backend.agents.state import AgentState
from backend.services.summary_service import generate_document_summary

logger = logging.getLogger(__name__)

def summary_agent(state: AgentState) -> dict:
    """
    Summary Agent Node:
    Summarizes the selected document(s) based on user query context.
    """
    selected_docs = state.get("selected_document_ids")
    query = state.get("query", "").lower()
    
    if not selected_docs:
        return {
            "summary_result": "No documents selected for summarization.",
            "next_agent": "citation_agent"
        }
        
    # Determine the type of summary requested by scanning the query
    summary_type = "detailed"
    if "short" in query or "brief" in query or "concise" in query:
        summary_type = "short"
    elif "bullet" in query or "list" in query or "points" in query:
        summary_type = "bullet"
        
    # Parse for specific page numbers in query, e.g., "page 3"
    page_num = None
    import re
    page_match = re.search(r"page\s+(\d+)", query)
    if page_match:
        try:
            page_num = int(page_match.group(1))
        except ValueError:
            pass
            
    summaries = []
    for doc_id in selected_docs:
        summary = generate_document_summary(
            document_id=doc_id,
            summary_type=summary_type,
            page_number=page_num
        )
        summaries.append(summary)
        
    final_summary_text = "\n\n---\n\n".join(summaries)
    
    # Store in summary_result, and transition to citation agent
    return {
        "summary_result": final_summary_text,
        "next_agent": "citation_agent"
    }
