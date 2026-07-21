import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from backend.config import settings

logger = logging.getLogger(__name__)

class ChromaService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        # Load local embedding model
        logger.info("Initializing SentenceTransformer all-MiniLM-L6-v2...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_assistant",
            metadata={"hnsw:space": "cosine"}  # Using Cosine similarity
        )
        logger.info("ChromaDB service initialized successfully.")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Adds text chunks and their metadata to ChromaDB.
        Generates embeddings locally using sentence-transformers.
        """
        if not chunks:
            return True
            
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            doc_id = chunk["metadata"]["document_id"]
            chunk_idx = chunk["metadata"]["chunk_index"]
            page_num = chunk["metadata"].get("page", 1)
            chunk_id = f"doc_{doc_id}_page_{page_num}_chunk_{chunk_idx}"
            
            ids.append(chunk_id)
            documents.append(chunk["text"])
            
            # Chroma DB metadata must be flat dictionaries of strings/ints/floats/bools
            metadatas.append({
                "document_id": int(doc_id),
                "filename": str(chunk["metadata"]["filename"]),
                "page": int(chunk["metadata"]["page"]),
                "upload_timestamp": str(chunk["metadata"]["upload_timestamp"]),
                "owner_id": int(chunk["metadata"]["owner_id"]),
                "chunk_index": int(chunk_idx)
            })
            
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(documents, show_progress_bar=False).tolist()
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise e

    def query(self, user_id: int, query_text: str, document_ids: Optional[List[int]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for the top-k relevant chunks.
        Filters by user_id to prevent cross-tenant queries, and optionally filters by document_ids.
        """
        try:
            # Embed query text
            query_embedding = self.model.encode([query_text], show_progress_bar=False).tolist()
            
            # Build filters
            # ChromaDB filters must be flat or nested using $and, $or
            where_filter = {"owner_id": int(user_id)}
            
            if document_ids:
                if len(document_ids) == 1:
                    where_filter = {
                        "$and": [
                            {"owner_id": int(user_id)},
                            {"document_id": int(document_ids[0])}
                        ]
                    }
                else:
                    where_filter = {
                        "$and": [
                            {"owner_id": int(user_id)},
                            {"document_id": {"$in": [int(d) for d in document_ids]}}
                        ]
                    }
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter
            )
            
            formatted_results = []
            if not results or not results["documents"] or not results["documents"][0]:
                return formatted_results
                
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for i in range(len(docs)):
                # Convert cosine distance to a confidence score
                # For cosine distance range [0, 2], similarity is 1 - distance/2
                # But typically cosine distance is [0, 1] for positive similarity. Let's do a safe conversion.
                dist = distances[i]
                sim = 1.0 - (dist / 2.0)
                confidence = max(0.0, min(1.0, sim))
                
                formatted_results.append({
                    "text": docs[i],
                    "metadata": metas[i],
                    "score": round(confidence, 4)
                })
                
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    def delete_document(self, document_id: int) -> bool:
        """
        Deletes all chunks associated with a specific document ID.
        """
        try:
            # Delete by where condition
            self.collection.delete(
                where={"document_id": int(document_id)}
            )
            logger.info(f"Successfully deleted chunks for document {document_id} from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document from ChromaDB: {e}")
            return False

    def update_document_filename(self, document_id: int, new_filename: str) -> bool:
        """
        Updates the filename metadata of all chunks associated with a document ID.
        """
        try:
            results = self.collection.get(
                where={"document_id": int(document_id)},
                include=["metadatas"]
            )
            if results and results["ids"]:
                ids = results["ids"]
                updated_metadatas = []
                for meta in results["metadatas"]:
                    meta["filename"] = str(new_filename)
                    updated_metadatas.append(meta)
                self.collection.update(ids=ids, metadatas=updated_metadatas)
                logger.info(f"Successfully updated filename in ChromaDB for document {document_id}.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating document filename in ChromaDB: {e}")
            return False

    def count_user_chunks(self, user_id: int) -> int:
        """
        Returns the total number of chunks uploaded by a specific user.
        """
        try:
            # We count by fetching metadatas for user
            results = self.collection.get(
                where={"owner_id": int(user_id)},
                include=["metadatas"]
            )
            if results and results["ids"]:
                return len(results["ids"])
            return 0
        except Exception as e:
            logger.error(f"Error counting user chunks: {e}")
            return 0

# Singleton instance
chroma_service = ChromaService()
