import re
import logging
from typing import Dict, List, Any
from backend.agents.state import AgentState
from backend.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def citation_agent(state: AgentState) -> dict:
    """
    Citation Agent Node:
    - Generates responses for RAG queries using retrieved chunks, enforcing inline citations [1], [2].
    - Extract citations (filename, page number, and confidence score) from the answer.
    - Resolves summaries/comparisons into final answers with document citations.
    """
    retrieved_context = state.get("retrieved_context", [])
    summary_result = state.get("summary_result", "")
    comparison_result = state.get("comparison_result", "")
    query = state.get("query", "")
    
    # Initialize output variables
    final_answer = ""
    citations = []
    
    # Case 1: Summarization workflow
    if summary_result:
        final_answer = summary_result
        # The citation is the document(s) summarized
        # We can try to extract from the selected document IDs or metadata
        if retrieved_context:
            # If retriever was run beforehand
            seen = set()
            for chunk in retrieved_context:
                meta = chunk["metadata"]
                key = (meta["filename"], meta.get("page", 1))
                if key not in seen:
                    seen.add(key)
                    citations.append({
                        "filename": meta["filename"],
                        "page": meta.get("page", 1),
                        "score": 1.0  # Summary is direct content, confidence is high
                    })
        else:
            # Default fallback when summary result is present
            citations.append({
                "filename": "Selected Document(s)",
                "page": 1,
                "score": 1.0
            })
            
    # Case 2: Comparison workflow
    elif comparison_result:
        final_answer = comparison_result
        citations.append({
            "filename": "Compared Documents",
            "page": 1,
            "score": 1.0
        })
        
    # Case 3: Retrieval/RAG workflow (Standard Q&A)
    else:
        if not retrieved_context:
            return {
                "final_answer": "I could not find any relevant information in the uploaded documents to answer your question.",
                "citations": []
            }
            
        # Format the chunks as context
        context_parts = []
        for i, chunk in enumerate(retrieved_context):
            meta = chunk["metadata"]
            context_parts.append(
                f"[{i + 1}] Document: {meta['filename']} | Page: {meta.get('page', 1)}\nContent: {chunk['text']}"
            )
        context_str = "\n\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an enterprise AI assistant answering user queries based ONLY on the provided context chunks. "
                "For every claim or piece of information you provide, you MUST cite the corresponding chunk number "
                "using square brackets (e.g. [1], [2]). You can cite multiple chunks if appropriate (e.g. [1][3]). "
                "If the context does not contain the answer, reply: 'I cannot find the answer in the uploaded documents.' "
                "Do NOT use any external knowledge. Stay completely objective and factual."
            )),
            ("user", "Context Chunks:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        ])
        
        try:
            llm = get_llm(temperature=0.0)
            chain = prompt | llm | StrOutputParser()
            
            raw_answer = chain.invoke({
                "context": context_str,
                "query": query
            })
            
            final_answer = raw_answer
            
            # Parse citations from raw answer
            # Look for patterns like [1], [2], [1][2], etc.
            citation_indices = re.findall(r"\[(\d+)\]", raw_answer)
            unique_indices = sorted(list(set(int(idx) for idx in citation_indices)))
            
            for idx in unique_indices:
                if 1 <= idx <= len(retrieved_context):
                    chunk = retrieved_context[idx - 1]
                    meta = chunk["metadata"]
                    citations.append({
                        "filename": meta["filename"],
                        "page": int(meta.get("page", 1)),
                        "score": float(chunk["score"])
                    })
        except Exception as e:
            logger.error(f"Error in RAG generation / citation parsing: {e}")
            final_answer = f"Failed to generate answer due to error: {str(e)}"
            citations = []
            
    return {
        "final_answer": final_answer,
        "citations": citations
    }
