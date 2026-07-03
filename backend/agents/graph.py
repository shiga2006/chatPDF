import logging
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.llm_factory import get_llm
from backend.agents.memory.agent import rewrite_query_with_history
from backend.agents.retriever.agent import retrieval_agent
from backend.agents.summary.agent import summary_agent
from backend.agents.comparison.agent import comparison_agent
from backend.agents.citation.agent import citation_agent
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def memory_node(state: AgentState) -> dict:
    """
    Memory Agent Node:
    Takes the recent conversation history and current user query,
    and rewrites it into a standalone query for better vector retrieval.
    """
    logger.info("Starting memory node processing...")
    messages = state.get("messages", [])
    query = state.get("query", "")
    
    # We want to separate history from current query.
    # The history contains messages up to the current one.
    history_messages = messages[:-1] if len(messages) > 1 else []
    
    # Rewrite the query
    rewritten_query = rewrite_query_with_history(history_messages, query)
    
    return {
        "query": rewritten_query
    }

def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor Agent Node:
    Decides which agent to execute next based on the rewritten query.
    """
    logger.info("Supervisor Agent analyzing routing...")
    query = state.get("query", "")
    query_lower = query.lower()
    
    # Rule-based fast paths
    if any(k in query_lower for k in ["compare", "comparison", "versus", " vs ", "difference", "similarities"]):
        next_agent = "comparison_agent"
    elif any(k in query_lower for k in ["summarize", "summary", "overview", "bullet points", "executive summary"]):
        next_agent = "summary_agent"
    else:
        # LLM classification
        prompt = (
            "You are the Supervisor Agent for an enterprise document knowledge assistant.\n"
            f"Analyze the user query: \"{query}\"\n\n"
            "Decide which of the following agents is best suited to answer it:\n"
            "- 'summary_agent': if the user explicitly wants a summary, recap, list of bullet points from a document or specific pages.\n"
            "- 'comparison_agent': if the user wants to compare multiple documents, find differences, similarities, or contrast rules.\n"
            "- 'retrieval_agent': for all other standard fact-finding, search, and informational questions about document contents.\n\n"
            "Respond ONLY with the name of the agent ('summary_agent', 'comparison_agent', or 'retrieval_agent') and nothing else. No punctuation."
        )
        try:
            llm = get_llm(temperature=0.0)
            response = llm.invoke([HumanMessage(content=prompt)])
            next_agent = response.content.strip().lower()
            
            # Clean up the output
            if "summary" in next_agent:
                next_agent = "summary_agent"
            elif "comparison" in next_agent or "compare" in next_agent:
                next_agent = "comparison_agent"
            else:
                next_agent = "retrieval_agent"
        except Exception as e:
            logger.error(f"Supervisor classification error: {e}. Defaulting to retrieval_agent.")
            next_agent = "retrieval_agent"
            
    logger.info(f"Supervisor routed to: {next_agent}")
    return {"next_agent": next_agent}

# Create LangGraph builder
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("memory_node", memory_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("summary_agent", summary_agent)
workflow.add_node("comparison_agent", comparison_agent)
workflow.add_node("citation_agent", citation_agent)

# Set Entry Point
workflow.set_entry_point("memory_node")

# Add edges
workflow.add_edge("memory_node", "supervisor")

# Conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "retrieval_agent": "retrieval_agent",
        "summary_agent": "summary_agent",
        "comparison_agent": "comparison_agent"
    }
)

# Route specialized agents to Citation Agent
workflow.add_edge("retrieval_agent", "citation_agent")
workflow.add_edge("summary_agent", "citation_agent")
workflow.add_edge("comparison_agent", "citation_agent")

# Citation Agent finishes the run
workflow.add_edge("citation_agent", END)

# Compile the LangGraph
agent_graph = workflow.compile()
