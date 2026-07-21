import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def split_text(self, text: str) -> List[str]:
            if not text:
                return []

            chunks = []
            start = 0
            text_length = self.length_function(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunks.append(text[start:end])
                if end >= text_length:
                    break
                start = max(0, end - self.chunk_overlap)

            return chunks

def extract_and_chunk_pdf(filepath: str, document_id: int, filename: str, owner_id: int) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF page-by-page and splits it into chunks.
    Keeps track of exact page numbers for Citations.
    
    Args:
        filepath: Absolute path to the PDF file.
        document_id: Database ID of the document.
        filename: Original file name.
        owner_id: User ID of the document owner.
        
    Returns:
        A list of dictionaries containing 'text' and 'metadata'.
    """
    doc = fitz.open(filepath)
    
    # Text splitter with configured sizes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    global_chunk_index = 0
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        
        # Skip empty pages
        if not page_text.strip():
            continue
            
        page_chunks = text_splitter.split_text(page_text)
        
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document_id": int(document_id),
                    "filename": str(filename),
                    "page": int(page_num + 1),  # 1-based index
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "owner_id": int(owner_id),
                    "chunk_index": global_chunk_index,
                    "page_chunk_index": i
                }
            })
            global_chunk_index += 1
            
    return chunks
