import logging
import re
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

DOMAIN_KEYWORDS = {
    "hr": ["hr", "benefit", "benefits", "leave", "policy", "employee", "payroll", "holiday", "vacation"],
    "finance": ["finance", "invoice", "budget", "expense", "cost", "payment", "billing", "tax", "reimbursement"],
    "it": ["it", "system", "access", "network", "password", "software", "application", "security", "device"],
    "product": ["product", "roadmap", "feature", "release", "spec", "requirements", "launch", "version"],
}

def _select_domain(query: str) -> str:
    query_lower = query.lower()
    best_domain = "general"
    best_score = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain

def _split_subtasks(query: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\b(?:and then|then|and also|also)\b|;|\n", query) if part.strip()]
    if len(parts) <= 1:
        return [query.strip()] if query.strip() else []
    return parts[:4]

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
        "query": rewritten_query,
        "normalized_query": rewritten_query
    }

def coarse_router_node(state: AgentState) -> dict:
    """
    Stage 3a: Coarse domain routing.
    Chooses a broad document domain and classifies the request shape.
    """
    query = state.get("query", "")
    domain = _select_domain(query)
    query_lower = query.lower()

    if any(k in query_lower for k in ["compare", "comparison", "versus", " vs ", "difference", "similarities"]):
        route_class = "comparison"
    elif any(k in query_lower for k in ["summarize", "summary", "overview", "bullet points", "executive summary"]):
        route_class = "summary"
    else:
        route_class = "retrieval"

    return {
        "selected_domain": domain,
        "next_agent": route_class
    }

def planner_node(state: AgentState) -> dict:
    """
    Stage 4a: Break multi-part questions into subtasks for multi-hop retrieval.
    """
    query = state.get("query", "")
    route_class = state.get("next_agent", "retrieval")

    if route_class != "retrieval":
        subtasks = [query.strip()] if query.strip() else []
    else:
        subtasks = _split_subtasks(query)

    return {
        "subtasks": subtasks,
        "selected_tools": [
            "semantic_retrieval" if route_class == "retrieval" else route_class,
            "citation",
            "self_verification"
        ]
    }

def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor Agent Node:
    Decides which agent to execute next based on the rewritten query,
    selected domain, and planned subtasks.
    """
    logger.info("Supervisor Agent analyzing routing...")
    query = state.get("query", "")
    selected_domain = state.get("selected_domain", "general")
    subtasks = state.get("subtasks", [])
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
            
    logger.info(f"Supervisor routed to: {next_agent} for domain={selected_domain} with {len(subtasks)} subtasks")
    return {
        "next_agent": next_agent,
        "selected_tools": state.get("selected_tools", [])
    }

def verification_node(state: AgentState) -> dict:
    """
    Stage 5: Evaluate retrieval quality, source coverage, and answer grounding.
    Retries once with a broader query if confidence is too low.
    """
    retrieved_context = state.get("retrieved_context", [])
    citations = state.get("citations", [])
    final_answer = state.get("final_answer", "")
    attempts = state.get("retrieval_attempts", 0)

    context_scores = [float(chunk.get("score", 0.0)) for chunk in retrieved_context if isinstance(chunk, dict)]
    avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
    citation_coverage = len(citations) / max(1, len(retrieved_context)) if retrieved_context else 0.0

    answer_penalty = 0.0
    if not final_answer.strip() or "cannot find the answer" in final_answer.lower():
        answer_penalty = 0.35

    verification_score = max(0.0, min(1.0, (avg_context_score * 0.6) + (citation_coverage * 0.4) - answer_penalty))
    needs_reretrieval = verification_score < 0.45 and attempts < 1

    if needs_reretrieval:
        retry_query = state.get("normalized_query") or state.get("query", "")
        return {
            "verification_score": round(verification_score, 3),
            "verification_notes": "Low grounding confidence; rerunning retrieval once with the normalized query.",
            "needs_reretrieval": True,
            "query": retry_query,
            "retrieval_attempts": attempts + 1,
            "next_agent": "retrieval_agent"
        }

    return {
        "verification_score": round(verification_score, 3),
        "verification_notes": "Answer grounded sufficiently in retrieved sources.",
        "needs_reretrieval": False,
        "next_agent": "final"
    }

# Create LangGraph builder
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("memory_node", memory_node)
workflow.add_node("coarse_router", coarse_router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("summary_agent", summary_agent)
workflow.add_node("comparison_agent", comparison_agent)
workflow.add_node("citation_agent", citation_agent)
workflow.add_node("verification_node", verification_node)

# Set Entry Point
workflow.set_entry_point("memory_node")

# Add edges
workflow.add_edge("memory_node", "coarse_router")
workflow.add_edge("coarse_router", "planner")
workflow.add_edge("planner", "supervisor")

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

# Citation Agent goes to verification
workflow.add_edge("citation_agent", "verification_node")

# Verification either ends the run or retries retrieval once
workflow.add_conditional_edges(
    "verification_node",
    lambda state: "retrieval_agent" if state.get("needs_reretrieval") else END,
    {
        "retrieval_agent": "retrieval_agent",
        END: END
    }
)

# Compile the LangGraph
agent_graph = workflow.compile()
