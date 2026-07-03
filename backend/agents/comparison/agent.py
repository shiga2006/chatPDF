import logging
from typing import Dict, List, Any
from backend.agents.state import AgentState
from backend.vectorstore.chroma_service import chroma_service
from backend.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def comparison_agent(state: AgentState) -> dict:
    """
    Comparison Agent Node:
    Compares two or more documents and generates a comparison markdown table.
    """
    selected_docs = state.get("selected_document_ids")
    user_id = state.get("user_id")
    query = state.get("query", "")
    
    if not selected_docs or len(selected_docs) < 2:
        return {
            "comparison_result": (
                "⚠️ Comparison requires at least 2 documents to be selected. "
                "Please select multiple documents from the sidebar and try again."
            ),
            "next_agent": "citation_agent"
        }
        
    try:
        # Load content for each document
        documents_data = {}
        for doc_id in selected_docs:
            results = chroma_service.collection.get(
                where={"document_id": int(doc_id)},
                include=["documents", "metadatas"]
            )
            
            if not results or not results["documents"]:
                continue
                
            # Reconstruct text
            chunks = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                chunks.append((meta.get("chunk_index", 0), doc))
            chunks.sort(key=lambda x: x[0])
            
            # Limit each document's text to fit context (~12,000 characters per document)
            filename = results["metadatas"][0].get("filename", f"Doc {doc_id}")
            doc_text = "\n".join([text for _, text in chunks])[:12000]
            documents_data[filename] = doc_text
            
        if not documents_data or len(documents_data) < 2:
            return {
                "comparison_result": "Could not retrieve sufficient content for the selected documents to perform comparison.",
                "next_agent": "citation_agent"
            }
            
        # Format documents content for LLM
        formatted_docs = ""
        for name, text in documents_data.items():
            formatted_docs += f"=== DOCUMENT: {name} ===\n{text}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert enterprise policy analyst. Compare the provided documents. "
                "Identify key parameters (such as eligibility, limits, coverage, dates, or terms) "
                "and compare them side-by-side. Your final response MUST include a detailed Markdown "
                "comparison table highlighting the differences and similarities. "
                "Strictly base your answers on the provided document texts. Do not make assumptions or inject external knowledge."
            )),
            ("user", "Compare the following documents based on the request: {query}\n\n{documents_content}")
        ])
        
        llm = get_llm(temperature=0.1)
        chain = prompt | llm | StrOutputParser()
        
        comparison = chain.invoke({
            "query": query,
            "documents_content": formatted_docs
        })
        
        return {
            "comparison_result": comparison,
            "next_agent": "citation_agent"
        }
        
    except Exception as e:
        logger.error(f"Error in Comparison Agent: {e}")
        return {
            "comparison_result": f"Failed to compare documents due to error: {str(e)}",
            "next_agent": "citation_agent"
        }
