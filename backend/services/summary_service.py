import logging
from typing import Optional, List
import re
from backend.vectorstore.chroma_service import chroma_service
from backend.agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _build_extractive_summary(full_text: str, summary_type: str, filename: str, page_number: Optional[int]) -> str:
    cleaned_text = re.sub(r"\s+", " ", full_text).strip()
    sentences = _split_sentences(cleaned_text)

    if not sentences:
        return f"No content found to summarize for {filename}."

    if summary_type == "short":
        selected = sentences[:2]
        body = " ".join(selected)
        return f"## Executive Summary\n\n{body}"

    if summary_type == "bullet":
        selected = sentences[:8]
        bullet_lines = [f"- {sentence}" for sentence in selected]
        return "## Key Points\n\n" + "\n".join(bullet_lines)

    selected = sentences[:10]
    sections = [
        "## Overview",
        "\n".join(f"- {sentence}" for sentence in selected[:4]),
    ]

    if len(selected) > 4:
        sections.extend([
            "\n## Additional Details",
            "\n".join(f"- {sentence}" for sentence in selected[4:10]),
        ])

    summary = "\n\n".join(sections)

    if page_number is not None:
        summary += f"\n\n_Source: Page {page_number}_"

    return summary

def generate_document_summary(document_id: int, summary_type: str = "detailed", page_number: Optional[int] = None) -> str:
    """
    Retrieves document chunks from ChromaDB and generates a summary using the configured LLM.
    
    Args:
        document_id: ID of the document.
        summary_type: Type of summary: 'short', 'detailed', or 'bullet'.
        page_number: Optional page number to restrict summary to.
    """
    try:
        # Retrieve chunks from ChromaDB for this document
        where_cond = {"document_id": int(document_id)}
        if page_number is not None:
            # If a page is specified, filter by that page
            where_cond = {
                "$and": [
                    {"document_id": int(document_id)},
                    {"page": int(page_number)}
                ]
            }
            
        results = chroma_service.collection.get(
            where=where_cond,
            include=["documents", "metadatas"]
        )
        
        if not results or not results["documents"]:
            return f"No content found for Document ID {document_id}."
            
        # Reconstruct the text in index order
        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append((meta.get("chunk_index", 0), doc, meta.get("page", 1)))
            
        chunks.sort(key=lambda x: x[0])
        
        # Limit text content size to ~25,000 characters to fit well within local LLM context limits
        full_text_parts = []
        total_len = 0
        filename = results["metadatas"][0].get("filename", f"Document {document_id}")
        
        for idx, text, page in chunks:
            if total_len + len(text) > 25000:
                break
            full_text_parts.append(f"[Page {page}] {text}")
            total_len += len(text)
            
        full_text = "\n\n".join(full_text_parts)
        
        # Determine prompt based on summary type
        if summary_type == "short":
            prompt_instruction = "Generate a concise, 1-2 paragraph executive summary of this document."
        elif summary_type == "bullet":
            prompt_instruction = "Generate a bulleted summary highlighting the most critical policies, dates, and instructions in this document. Use clean markdown bullet points."
        else:  # detailed
            prompt_instruction = "Generate a detailed, structured summary of this document. Include sections with clear markdown headings (e.g., Overview, Key Policies, Major Dates/Timelines, Important Notes)."
            
        if page_number is not None:
            prompt_instruction += f" Note: This summary should ONLY focus on the content of Page {page_number}."
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert enterprise knowledge analyst. Summarize the provided document text "
                "following the instructions below. ONLY use the provided text. Do not invent or assume anything. "
                "If the text is empty or lacks information, state that clearly.\n\n"
                f"Instruction: {prompt_instruction}"
            )),
            ("user", "Document: {filename}\n\nContent:\n{content}")
        ])
        
        try:
            llm = get_llm(temperature=0.2)
            chain = prompt | llm | StrOutputParser()

            summary = chain.invoke({
                "filename": filename,
                "content": full_text
            })

            return summary
        except Exception as llm_error:
            logger.warning(
                "LLM summary generation failed for document %s, using extractive fallback: %s",
                document_id,
                llm_error,
            )
            return _build_extractive_summary(full_text, summary_type, filename, page_number)
        
    except Exception as e:
        logger.error(f"Error generating document summary: {e}")
        return f"Failed to generate summary due to error: {str(e)}"
