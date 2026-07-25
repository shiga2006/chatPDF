"""
MCP Tool Clustering & Routing.

Implements a two-level clustering approach:
1. Tool-level clustering (offline): Groups agent tools by capability similarity
   using K-Means on tool descriptions. This creates tool clusters like
   "retrieval tools", "summary tools", etc.

2. Query-level routing (online): Embeds incoming user queries and maps them
   to the nearest tool cluster (or specific tool) using cluster centroids.

Benefits over the original hardcoded router:
- Automatically adapts to new tools (just register them)
- More nuanced routing via semantic similarity
- Confidence scores for routing decisions
- Fallback mechanism when confidence is low
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer

from backend.agents.mcp_routing.agent_tools import (
    AgentTool,
    AgentToolRegistry,
    agent_tool_registry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Result of routing a query to a tool or tool cluster."""

    tool_name: str
    category: str
    confidence: float
    cluster_id: int
    cluster_label: str
    is_fallback: bool = False
    matched_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "category": self.category,
            "confidence": self.confidence,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "is_fallback": self.is_fallback,
            "matched_keywords": self.matched_keywords,
        }


# ---------------------------------------------------------------------------
# MCP Tool Router with K-Means Clustering
# ---------------------------------------------------------------------------


class MCPToolRouter:
    """
    Routes user queries to the best MCP tool using K-Means clustering.

    Two-stage routing:
    Stage 1: Cluster tools by their descriptions → discover tool groups.
    Stage 2: Map user queries to the nearest tool cluster → select best tool.

    Can operate in two modes:
    - 'cluster': Route to the best tool cluster (category) and let the
      supervisor pick the specific agent.
    - 'tool': Route directly to the most similar specific tool.
    """

    def __init__(
        self,
        registry: AgentToolRegistry,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        n_tool_clusters: int = 5,
        routing_mode: str = "cluster",
        confidence_threshold: float = 0.35,
        random_state: int = 42,
    ):
        self.registry = registry
        self.embedding_model_name = embedding_model_name
        self.n_tool_clusters = n_tool_clusters
        self.routing_mode = routing_mode
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

        # Embedding model
        logger.info(f"Loading embedding model '{embedding_model_name}' for tool routing...")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # K-Means for tool clustering
        self.tool_kmeans: Optional[KMeans] = None
        self.tool_cluster_map: Dict[int, List[str]] = {}  # cluster_id -> [tool_names]
        self.tool_embeddings: Optional[np.ndarray] = None
        self.tool_names: List[str] = []
        self.tool_texts: List[str] = []
        self.is_trained = False

        # Keyword-based fallback patterns
        self._keyword_patterns: Dict[str, List[str]] = {
            "retrieval": [
                "what", "tell", "find", "search", "lookup", "where", "when", "who",
                "how many", "how much", "is there", "are there", "does",
                "explain", "describe", "give me",
            ],
            "summary": [
                "summarize", "summary", "overview", "recap", "bullet point",
                "brief", "condense", "tl;dr", "executive summary",
            ],
            "comparison": [
                "compare", "comparison", "versus", " vs ", "difference",
                "similarity", "contrast", "both", "either",
            ],
            "citation": [
                "cite", "reference", "source", "attribution", "where did you get",
            ],
            "verification": [
                "verify", "check", "confirm", "validate", "accurate", "correct",
                "is this right", "double check",
            ],
        }

    # ------------------------------------------------------------------
    # Training / Clustering Tools
    # ------------------------------------------------------------------

    def fit_tool_clusters(self) -> "MCPToolRouter":
        """
        Cluster registered tools by their description embeddings.
        This creates semantic groups of related capabilities.
        """
        tool_texts_dict = self.registry.get_tool_embeddings_text()
        self.tool_names = list(tool_texts_dict.keys())
        self.tool_texts = list(tool_texts_dict.values())

        if len(self.tool_names) < 2:
            logger.warning("Fewer than 2 tools registered; skipping tool clustering.")
            return self

        # Embed tool descriptions
        embeddings = self.embedding_model.encode(self.tool_texts, show_progress_bar=False)
        self.tool_embeddings = normalize(np.array(embeddings), axis=1, norm="l2")

        # K-Means on tools
        n = min(self.n_tool_clusters, len(self.tool_names))
        self.tool_kmeans = KMeans(
            n_clusters=n,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = self.tool_kmeans.fit_predict(self.tool_embeddings)

        # Build cluster → tool mapping
        self.tool_cluster_map = {}
        for tool_name, label in zip(self.tool_names, labels):
            label_int = int(label)
            if label_int not in self.tool_cluster_map:
                self.tool_cluster_map[label_int] = []
            self.tool_cluster_map[label_int].append(tool_name)

        self.is_trained = True

        # Log cluster composition
        for cid, tools in self.tool_cluster_map.items():
            logger.info(f"Tool Cluster {cid}: {', '.join(tools)}")

        return self

    # ------------------------------------------------------------------
    # Keyword-Based Fallback Classification
    # ------------------------------------------------------------------

    def _classify_by_keywords(self, query: str) -> Tuple[str, float, List[str]]:
        """
        Fallback: classify query by keyword matching.

        Returns:
            Tuple of (category_name, confidence, matched_keywords)
        """
        query_lower = query.lower()
        best_category = "retrieval"
        best_score = 0
        best_matches: List[str] = []

        for category, patterns in self._keyword_patterns.items():
            score = 0
            matches: List[str] = []
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
                    matches.append(pattern)
            if score > best_score:
                best_score = score
                best_category = category
                best_matches = matches

        # Normalize confidence
        confidence = min(1.0, best_score / 3.0)
        return best_category, confidence, best_matches

    # ------------------------------------------------------------------
    # Online Routing
    # ------------------------------------------------------------------

    def route_query(
        self,
        query: str,
        mode: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route a user query to the best tool or tool category.

        Uses a cascading approach:
        1. Try K-Means clustering (if trained) → high-confidence route
        2. Fall back to keyword matching → medium-confidence route
        3. Ultimate fallback → default retrieval tool

        Args:
            query: The user's question/query.
            mode: Override routing mode ('cluster' or 'tool').

        Returns:
            RoutingDecision with the best tool/category match.
        """
        mode = mode or self.routing_mode
        query_lower = query.lower()

        # --- Attempt 1: K-Means clustering routing ---
        if self.is_trained and self.tool_kmeans is not None and self.tool_embeddings is not None:
            query_emb = self.embedding_model.encode([query], show_progress_bar=False)
            query_emb = normalize(np.array(query_emb), axis=1, norm="l2")

            # Find nearest cluster centroid
            distances = self.tool_kmeans.transform(query_emb)[0]
            nearest_cluster = int(np.argmin(distances))
            confidence = float(np.exp(-distances[nearest_cluster]))
            confidence = max(0.0, min(1.0, confidence))

            if confidence >= self.confidence_threshold and nearest_cluster in self.tool_cluster_map:
                tools_in_cluster = self.tool_cluster_map[nearest_cluster]
                # Find the best individual tool within this cluster
                tool_dists = np.linalg.norm(
                    self.tool_embeddings - query_emb, axis=1
                )
                best_tool_idx = int(np.argmin(tool_dists))
                best_tool_name = self.tool_names[best_tool_idx]
                best_tool = self.registry.get(best_tool_name)

                if best_tool:
                    return RoutingDecision(
                        tool_name=best_tool_name,
                        category=best_tool.category,
                        confidence=confidence,
                        cluster_id=nearest_cluster,
                        cluster_label=f"tool_cluster_{nearest_cluster}",
                        matched_keywords=[],
                    )

                # If no specific tool, route to the category
                if mode == "cluster" and tools_in_cluster:
                    representative_tool = self.registry.get(tools_in_cluster[0])
                    if representative_tool:
                        return RoutingDecision(
                            tool_name=representative_tool.name,
                            category=representative_tool.category,
                            confidence=confidence,
                            cluster_id=nearest_cluster,
                            cluster_label=f"tool_cluster_{nearest_cluster}",
                        )

        # --- Attempt 2: Keyword fallback ---
        category, kw_confidence, keywords = self._classify_by_keywords(query)

        # Map category to a representative tool
        category_tool_map = self._get_category_tool_map()
        if category in category_tool_map:
            tool_name = category_tool_map[category]
            return RoutingDecision(
                tool_name=tool_name,
                category=category,
                confidence=kw_confidence,
                cluster_id=-1,
                cluster_label="keyword_fallback",
                is_fallback=True,
                matched_keywords=keywords,
            )

        # --- Attempt 3: Ultimate fallback ---
        return RoutingDecision(
            tool_name="retrieve_semantic",
            category="retrieval",
            confidence=0.2,
            cluster_id=-1,
            cluster_label="ultimate_fallback",
            is_fallback=True,
            matched_keywords=[],
        )

    def batch_route(
        self, queries: List[str]
    ) -> List[RoutingDecision]:
        """Route multiple queries at once."""
        return [self.route_query(q) for q in queries]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_category_tool_map(self) -> Dict[str, str]:
        """Map each category to its highest-priority tool name."""
        tools = self.registry.list_tools()
        category_map: Dict[str, str] = {}
        for tool in tools:
            if tool.category not in category_map:
                category_map[tool.category] = tool.name
            else:
                # Keep higher priority
                existing = self.registry.get(category_map[tool.category])
                if existing and tool.priority > existing.priority:
                    category_map[tool.category] = tool.name
        return category_map

    def get_tool_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about tool clusters."""
        stats: Dict[str, Any] = {
            "is_trained": self.is_trained,
            "n_clusters": len(self.tool_cluster_map),
            "n_tools": len(self.tool_names),
            "routing_mode": self.routing_mode,
            "confidence_threshold": self.confidence_threshold,
        }
        if self.is_trained:
            stats["clusters"] = {
                str(cid): tools for cid, tools in self.tool_cluster_map.items()
            }
        return stats


# ---------------------------------------------------------------------------
# Convenience function for LangGraph integration
# ---------------------------------------------------------------------------

def route_via_mcp_clustering(
    query: str,
    router: Optional[MCPToolRouter] = None,
) -> str:
    """
    Route a query using MCP tool clustering and return the target agent name.

    This is the primary integration point with the LangGraph supervisor.

    Args:
        query: User query text.
        router: MCPToolRouter instance (uses global singleton if None).

    Returns:
        Agent node name: 'retrieval_agent', 'summary_agent', 'comparison_agent',
        'citation_agent', 'memory_agent', or 'verification_node'.
    """
    router = router or mcp_tool_router
    decision = router.route_query(query)

    # Map MCP tool categories to LangGraph agent nodes
    category_to_agent = {
        "retrieval": "retrieval_agent",
        "summary": "summary_agent",
        "comparison": "comparison_agent",
        "citation": "citation_agent",
        "memory": "memory_node",
        "verification": "verification_node",
    }

    agent = category_to_agent.get(decision.category, "retrieval_agent")
    logger.info(
        f"MCP Clustering Router: query='{query[:50]}...' → "
        f"category={decision.category}, tool={decision.tool_name}, "
        f"confidence={decision.confidence:.3f}, fallback={decision.is_fallback}"
    )
    return agent


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

mcp_tool_router = MCPToolRouter(
    registry=agent_tool_registry,
    embedding_model_name="all-MiniLM-L6-v2",
    n_tool_clusters=5,
    routing_mode="cluster",
    confidence_threshold=0.35,
)

# Auto-fit tool clusters on import
try:
    mcp_tool_router.fit_tool_clusters()
    logger.info("MCP Tool Router initialized and tool clusters fitted.")
except Exception as e:
    logger.warning(f"Could not fit tool clusters on import: {e}. Will retry at runtime.")

