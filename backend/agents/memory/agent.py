import logging
from typing import List, Dict, Any
from backend.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

def rewrite_query_with_history(messages: List[BaseMessage], current_query: str) -> str:
    """
    Rewrites the user's current query into a standalone query using the chat history.
    If the history is empty or the query is already standalone, it returns the current query.
    """
    # If no history, no need to rewrite
    if not messages:
        return current_query
        
    # Format history into a readable format for the LLM
    history_str = ""
    for msg in messages[-6:]:  # Consider last 6 messages
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a conversation context analyzer. Given a conversation history and a follow-up query, "
            "determine if the follow-up query depends on the previous context. "
            "If it depends on the context, rewrite it into a single, standalone query that includes all necessary details "
            "for a search database. If it is already a standalone question, return the user query exactly as it is.\n\n"
            "Do NOT answer the question. Just output the standalone query and nothing else."
        )),
        ("user", "Conversation History:\n{history}\n\nFollow-up Query: {query}\n\nStandalone Query:")
    ])
    
    try:
        llm = get_llm(temperature=0.0)
        chain = prompt | llm
        
        response = chain.invoke({
            "history": history_str,
            "query": current_query
        })
        
        rewritten = response.content.strip()
        logger.info(f"Original query: '{current_query}' -> Rewritten standalone query: '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"Error in query rewriting: {e}. Using original query.")
        return current_query
