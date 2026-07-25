"""
K-Means Clustering Module for Query Intent & Document Topic Clustering.

Provides:
1. Query Intent Clustering — Groups user queries into semantic clusters
   (e.g., policy, factual, procedural, numeric) to replace keyword-based domain routing.
2. Document Chunk Clustering — Groups document chunks into topic clusters
   for diversified retrieval (MMR-style diversification).
3. Adaptive Re-clustering — Periodically re-trains on accumulated queries
   to adapt to new patterns.

Uses sentence-transformers embeddings + sklearn KMeans.
"""

import logging
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default cluster definitions for query intents
# ---------------------------------------------------------------------------
QUERY_INTENT_LABELS: Dict[int, str] = {
    0: "policy_search",       # Looking up rules, policies, guidelines
    1: "factual_qa",          # Simple factual Q&A from documents
    2: "procedural_howto",    # How-to, step-by-step, process
    3: "numeric_analytics",   # Numbers, stats, limits, dates, amounts
    4: "comparison",          # Comparing multiple documents/items
    5: "summarization",       # Summarize, overview, executive summary
    6: "definitional",        # What is X? Define/explain term
}

# Default cluster descriptions for tool routing
QUERY_INTENT_DESCRIPTIONS: Dict[int, str] = {
    0: "Searching for specific policies, rules, or guidelines in documents",
    1: "Answering straightforward factual questions from document content",
    2: "Step-by-step procedures, how-to instructions, or workflows",
    3: "Questions involving numbers, statistics, limits, dates, or amounts",
    4: "Comparing two or more documents, policies, or items side-by-side",
    5: "Generating summaries, overviews, bullet points of documents",
    6: "Defining or explaining terms, concepts, or acronyms",
}


class EmbeddingCluster:
    """
    K-Means clustering over sentence-transformer embeddings.

    Supports both query intent clustering and document chunk topic clustering.
    Uses MiniBatchKMeans for efficiency with large datasets.
    """

    def __init__(
        self,
        n_clusters: int = 7,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
        use_minibatch: bool = True,
        persist_path: Optional[str] = None,
    ):
        self.n_clusters = n_clusters
        self.embedding_model_name = embedding_model_name
        self.random_state = random_state
        self.use_minibatch = use_minibatch
        self.persist_path = persist_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "chromadb", "cluster_model.pkl"
        )

        # Load embedding model
        logger.info(
            f"Loading embedding model '{embedding_model_name}' for clustering..."
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # K-Means model (lazy init)
        self.kmeans: Optional[KMeans] = None
        self.is_trained = False
        self.label_map: Dict[int, str] = dict(QUERY_INTENT_LABELS)
        self.label_descriptions: Dict[int, str] = dict(QUERY_INTENT_DESCRIPTIONS)

        # Stored embeddings for incremental fitting
        self._stored_embeddings: List[np.ndarray] = []
        self._stored_texts: List[str] = []

        # Try to load persisted model
        self._try_load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_path_exists(self) -> bool:
        return os.path.exists(self.persist_path)

    def _try_load(self) -> bool:
        """Attempt to load a previously trained K-Means model from disk."""
        try:
            if self._persist_path_exists():
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                self.kmeans = data.get("kmeans")
                self.is_trained = data.get("is_trained", False)
                self.n_clusters = data.get("n_clusters", self.n_clusters)
                self.label_map = data.get("label_map", self.label_map)
                self.label_descriptions = data.get(
                    "label_descriptions", self.label_descriptions
                )
                if self.is_trained and self.kmeans is not None:
                    logger.info(
                        f"Loaded pre-trained K-Means model from {self.persist_path}"
                    )
                    return True
        except Exception as e:
            logger.warning(f"Could not load clustering model from disk: {e}")
        return False

    def save(self) -> None:
        """Persist the trained K-Means model to disk."""
        if self.kmeans is None:
            logger.warning("No trained model to save.")
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "wb") as f:
                pickle.dump(
                    {
                        "kmeans": self.kmeans,
                        "is_trained": self.is_trained,
                        "n_clusters": self.n_clusters,
                        "label_map": self.label_map,
                        "label_descriptions": self.label_descriptions,
                        "embedding_model_name": self.embedding_model_name,
                        "trained_at": datetime.utcnow().isoformat(),
                    },
                    f,
                )
            logger.info(f"Clustering model saved to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save clustering model: {e}")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """Convert a list of texts to normalized embeddings."""
        if not texts:
            return np.array([])
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings)
        # Normalize for cosine-similarity-friendly clustering
        embeddings = normalize(embeddings, axis=1, norm="l2")
        return embeddings

    # ------------------------------------------------------------------
    # Training / Fitting
    # ------------------------------------------------------------------

    def fit(self, texts: List[str], n_clusters: Optional[int] = None) -> "EmbeddingCluster":
        """
        Fit K-Means on the provided texts.

        Args:
            texts: List of text strings to cluster.
            n_clusters: Override number of clusters (optional).

        Returns:
            self for chaining.
        """
        if not texts:
            logger.warning("No texts provided for clustering training.")
            return self

        if n_clusters is not None:
            self.n_clusters = n_clusters

        embeddings = self.embed(texts)

        # Auto-determine optimal clusters if not specified
        if self.n_clusters is None or self.n_clusters <= 1:
            self.n_clusters = self._auto_determine_clusters(embeddings)

        n = min(self.n_clusters, len(texts))
        logger.info(
            f"Fitting K-Means with {n} clusters on {len(texts)} texts..."
        )

        if self.use_minibatch:
            self.kmeans = MiniBatchKMeans(
                n_clusters=n,
                random_state=self.random_state,
                batch_size=min(256, len(texts)),
                n_init="auto",
            )
        else:
            self.kmeans = KMeans(
                n_clusters=n,
                random_state=self.random_state,
                n_init="auto",
            )

        self.kmeans.fit(embeddings)
        self.is_trained = True

        # Compute silhouette score (if enough samples)
        if len(texts) > n and n > 1:
            try:
                sil = silhouette_score(embeddings, self.kmeans.labels_)
                logger.info(f"Silhouette score: {sil:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute silhouette score: {e}")

        # Store
        self._stored_embeddings.append(embeddings)
        self._stored_texts.extend(texts)

        self.save()
        return self

    def incremental_fit(
        self, texts: List[str], max_samples: int = 5000
    ) -> "EmbeddingCluster":
        """
        Incrementally update the cluster model with new texts.
        Retrains from scratch on accumulated samples (up to max_samples).

        Args:
            texts: New texts to incorporate.
            max_samples: Maximum stored samples to retrain on.

        Returns:
            self for chaining.
        """
        self._stored_texts.extend(texts)
        # Trim to max_samples
        if len(self._stored_texts) > max_samples:
            self._stored_texts = self._stored_texts[-max_samples:]

        return self.fit(self._stored_texts)

    # ------------------------------------------------------------------
    # Prediction / Assignment
    # ------------------------------------------------------------------

    def predict(self, texts: List[str]) -> List[int]:
        """
        Assign cluster labels to texts.

        Args:
            texts: Texts to classify.

        Returns:
            List of cluster IDs (ints). Returns -1 if model not trained.
        """
        if not self.is_trained or self.kmeans is None:
            logger.warning("K-Means model not trained. Returning -1 for all.")
            return [-1] * len(texts)

        if not texts:
            return []

        embeddings = self.embed(texts)
        labels = self.kmeans.predict(embeddings)
        return labels.tolist()

    def predict_single(self, text: str) -> int:
        """Predict cluster for a single text."""
        labels = self.predict([text])
        return labels[0] if labels else -1

    def predict_with_confidence(self, text: str) -> Tuple[int, float]:
        """
        Predict cluster with confidence score (based on distance to centroid).

        Args:
            text: Single text to classify.

        Returns:
            Tuple of (cluster_id, confidence) where confidence is in [0, 1].
        """
        if not self.is_trained or self.kmeans is None:
            return -1, 0.0

        embedding = self.embed([text])
        # Get distances to centroids
        distances = self.kmeans.transform(embedding)[0]
        min_dist = np.min(distances)
        # Convert distance to confidence (lower distance = higher confidence)
        # Use a softmax-inspired normalization
        confidence = float(np.exp(-min_dist))
        confidence = max(0.0, min(1.0, confidence))
        cluster_id = int(np.argmin(distances))

        return cluster_id, confidence

    # ------------------------------------------------------------------
    # Cluster Info
    # ------------------------------------------------------------------

    def get_cluster_label(self, cluster_id: int) -> str:
        """Get human-readable label for a cluster."""
        return self.label_map.get(cluster_id, f"cluster_{cluster_id}")

    def get_cluster_description(self, cluster_id: int) -> str:
        """Get description of what a cluster represents."""
        return self.label_descriptions.get(
            cluster_id, f"Unknown cluster {cluster_id}"
        )

    def set_label_map(self, label_map: Dict[int, str]) -> None:
        """Override the default cluster label mapping."""
        self.label_map.update(label_map)

    def set_label_descriptions(self, descriptions: Dict[int, str]) -> None:
        """Override the default cluster description mapping."""
        self.label_descriptions.update(descriptions)

    def discover_labels_from_llm(
        self, cluster_samples: Dict[int, List[str]], llm_chain: Any = None
    ) -> None:
        """
        Use an LLM to discover human-readable labels for each cluster
        based on representative samples. Useful for adaptive labeling.

        Args:
            cluster_samples: Dict of {cluster_id: [sample_texts]}
            llm_chain: LangChain chain/prompt for labeling (optional).
        """
        # Simple heuristic: use top frequent words per cluster
        # This avoids an LLM dependency but can be enhanced later
        for cid, samples in cluster_samples.items():
            if not samples:
                continue
            # Concatenate samples and extract key terms
            combined = " ".join(samples)
            words = combined.lower().split()
            # Filter common stop-words
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "and",
                "or",
                "but",
                "not",
                "this",
                "that",
                "it",
                "from",
                "as",
                "by",
                "be",
                "has",
                "have",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
            }
            freq = {}
            for w in words:
                if w not in stop_words and len(w) > 2:
                    freq[w] = freq.get(w, 0) + 1
            top_terms = sorted(freq.items(), key=lambda x: -x[1])[:5]
            terms_str = ", ".join([t for t, _ in top_terms])
            self.label_map[cid] = f"topics_{terms_str}" if terms_str else f"cluster_{cid}"
            self.label_descriptions[
                cid
            ] = f"Cluster containing documents about: {terms_str}" if terms_str else f"Cluster {cid}"

        self.save()

    def get_cluster_centroids(self) -> Optional[np.ndarray]:
        """Get the cluster centroid embeddings."""
        if self.kmeans is not None:
            return self.kmeans.cluster_centers_
        return None

    # ------------------------------------------------------------------
    # Document Chunk Diversification
    # ------------------------------------------------------------------

    def diversify_retrieval(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        diversity_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved chunks to maximize topic diversity.
        Uses K-Means cluster assignments to ensure coverage across topics.

        Args:
            chunks: Retrieved chunks with 'text' and 'score'.
            top_k: Number of chunks to return.
            diversity_weight: Weight of diversity vs relevance (0 = pure relevance).

        Returns:
            Re-ranked list of chunks with diversity.
        """
        if not chunks or not self.is_trained:
            return chunks[:top_k]

        texts = [c["text"] for c in chunks]
        labels = self.predict(texts)

        # Group chunks by cluster
        cluster_groups: Dict[int, List[Tuple[int, Dict]]] = {}
        for idx, (label, chunk) in enumerate(zip(labels, chunks)):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append((idx, chunk))

        # Select top chunks, prioritizing underrepresented clusters
        selected: List[Dict] = []
        selected_indices: set = set()
        cluster_usage: Dict[int, int] = {k: 0 for k in cluster_groups}

        # Round-robin across clusters by relevance
        while len(selected) < top_k and any(
            len(items) > cluster_usage[cid]
            for cid, items in cluster_groups.items()
        ):
            for cid, items in cluster_groups.items():
                if len(selected) >= top_k:
                    break
                if cluster_usage[cid] < len(items):
                    idx, chunk = items[cluster_usage[cid]]
                    if idx not in selected_indices:
                        selected.append(chunk)
                        selected_indices.add(idx)
                        cluster_usage[cid] += 1

        # If we still need more, fill by original score
        if len(selected) < top_k:
            for chunk in chunks:
                if len(selected) >= top_k:
                    break
                if id(chunk) not in [id(s) for s in selected]:
                    selected.append(chunk)

        return selected[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auto_determine_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """
        Auto-determine optimal number of clusters using silhouette score.
        Evaluates cluster counts from 2 to max_clusters.
        """
        n_samples = len(embeddings)
        if n_samples < 3:
            return max(1, n_samples)

        max_k = min(max_clusters, n_samples - 1)
        best_k = 3  # default
        best_score = -1

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = km.fit_predict(embeddings)
            try:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        logger.info(f"Auto-determined optimal clusters: {best_k} (score={best_score:.4f})")
        return best_k

    def get_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        stats: Dict[str, Any] = {
            "is_trained": self.is_trained,
            "n_clusters": self.n_clusters,
            "embedding_model": self.embedding_model_name,
        }
        if self.kmeans is not None and self.is_trained:
            inertia = getattr(self.kmeans, "inertia_", None)
            if inertia is not None:
                stats["inertia"] = float(inertia)
            stats["n_features"] = (
                self.kmeans.cluster_centers_.shape[1]
                if hasattr(self.kmeans, "cluster_centers_")
                else 0
            )
            stats["persist_path"] = self.persist_path
        return stats


# ---------------------------------------------------------------------------
# Singleton instance for global use
# ---------------------------------------------------------------------------
cluster_manager = EmbeddingCluster(n_clusters=7)

