"""
MCP Agent Tool Definitions.

Defines all agent capabilities as MCP Tool objects for discovery and
clustering-based routing. Each tool wraps an agent's capability with
a name, description, and input schema.

Tools are organized by domain capability clusters:
- Retrieval tools (search, query)
- Summary tools (summarize, bullet points)
- Comparison tools (compare, contrast)
- Citation tools (cite, reference)
- Verification tools (check, validate)
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simplified MCP Tool definition (stdlib-compatible, no external deps needed)
# ---------------------------------------------------------------------------


@dataclass
class AgentTool:
    """
    Represents an agent capability as a routable MCP-style tool.

    Compatible with the MCP protocol 'Tool' type structure but defined
    locally to avoid coupling to a specific MCP library version.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    category: str  # e.g., "retrieval", "summary", "comparison", "citation", "verification"
    domain_tags: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = preferred when multiple tools match

    def to_mcp_dict(self) -> Dict[str, Any]:
        """Convert to MCP Tool-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "category": self.category,
            "domain_tags": self.domain_tags,
        }


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

# --- Retrieval Tools ---

TOOL_RETRIEVE_SEMANTIC = AgentTool(
    name="retrieve_semantic",
    description=(
        "Perform semantic vector search across uploaded documents to find "
        "the most relevant text chunks for a user's question. Returns chunks "
        "with metadata (filename, page, score). Best for factual Q&A."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to retrieve (default: 5)",
                "default": 5,
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional filter to specific document IDs",
            },
        },
        "required": ["query"],
    },
    category="retrieval",
    domain_tags=["search", "qa", "factual", "query"],
    priority=5,
)

TOOL_RETRIEVE_MULTI_HOP = AgentTool(
    name="retrieve_multi_hop",
    description=(
        "Multi-hop retrieval that breaks complex questions into subtasks "
        "and retrieves relevant chunks for each subtask. Best for multi-part "
        "or compound questions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The multi-part question"},
            "subtasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional pre-split subtasks",
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional document filter",
            },
        },
        "required": ["query"],
    },
    category="retrieval",
    domain_tags=["multi_hop", "compound", "complex", "subtask"],
    priority=4,
)

TOOL_RETRIEVE_DOMAIN = AgentTool(
    name="retrieve_domain_specific",
    description=(
        "Domain-specific retrieval filtered by document category "
        "(HR, Finance, IT, Product). Use when the query matches a "
        "specific business domain."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "domain": {
                "type": "string",
                "enum": ["hr", "finance", "it", "product", "general"],
                "description": "Document domain to search within",
            },
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query", "domain"],
    },
    category="retrieval",
    domain_tags=["domain", "filtered", "hr", "finance", "it", "product"],
    priority=3,
)

# --- Summary Tools ---

TOOL_SUMMARIZE_DOCUMENT = AgentTool(
    name="summarize_document",
    description=(
        "Generate a comprehensive summary of one or more documents. "
        "Supports detailed, short, and bullet-point formats."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Document IDs to summarize",
            },
            "summary_type": {
                "type": "string",
                "enum": ["detailed", "short", "bullet"],
                "description": "Type of summary to generate",
                "default": "detailed",
            },
            "page_number": {
                "type": "integer",
                "description": "Optional specific page to summarize",
            },
        },
        "required": ["document_ids"],
    },
    category="summary",
    domain_tags=["summary", "overview", "recap", "bullet"],
    priority=5,
)

TOOL_EXECUTIVE_SUMMARY = AgentTool(
    name="executive_summary",
    description=(
        "Generate a concise executive-level summary of documents. "
        "Focuses on key decisions, action items, and high-level takeaways."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Document IDs to summarize",
            },
        },
        "required": ["document_ids"],
    },
    category="summary",
    domain_tags=["executive", "high_level", "key_points"],
    priority=4,
)

# --- Comparison Tools ---

TOOL_COMPARE_DOCUMENTS = AgentTool(
    name="compare_documents",
    description=(
        "Compare two or more documents side-by-side, identifying "
        "differences and similarities across key parameters. "
        "Produces a markdown comparison table."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Document IDs to compare (minimum 2)",
                "minItems": 2,
            },
            "comparison_aspects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional specific aspects to compare",
            },
        },
        "required": ["document_ids"],
    },
    category="comparison",
    domain_tags=["compare", "contrast", "difference", "similarity"],
    priority=5,
)

# --- Citation Tools ---

TOOL_CITE_SOURCES = AgentTool(
    name="cite_sources",
    description=(
        "Format a response with inline citations [1], [2] referencing "
        "the source document chunks. Extracts filename, page number, "
        "and confidence scores for each citation."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "The response text to attach citations to",
            },
            "context_chunks": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Retrieved context chunks with metadata",
            },
        },
        "required": ["response", "context_chunks"],
    },
    category="citation",
    domain_tags=["citation", "reference", "source", "attribution"],
    priority=5,
)

# --- Verification Tools ---

TOOL_VERIFY_ANSWER = AgentTool(
    name="verify_answer",
    description=(
        "Verify the grounding quality of a generated answer by checking "
        "retrieval confidence scores and citation coverage. Can trigger "
        "a re-retrieval if confidence is below threshold."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The generated answer"},
            "retrieved_context": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Context chunks used for the answer",
            },
            "citations": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Citations attached to the answer",
            },
        },
        "required": ["answer"],
    },
    category="verification",
    domain_tags=["verify", "confidence", "grounding", "quality"],
    priority=5,
)

# --- Memory Tools ---

TOOL_REWRITE_QUERY = AgentTool(
    name="rewrite_query",
    description=(
        "Rewrite a user query with conversation history context "
        "to produce a standalone, self-contained query for retrieval."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The current user query"},
            "history": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Recent conversation history messages",
            },
        },
        "required": ["query"],
    },
    category="memory",
    domain_tags=["rewrite", "contextualize", "standalone"],
    priority=4,
)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class AgentToolRegistry:
    """
    Registry of all available AgentTools, organized by category.
    Supports discovery (list_tools), lookup by name, and category-based
    grouping for cluster routing.
    """

    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default agent tools."""
        for tool in [
            TOOL_RETRIEVE_SEMANTIC,
            TOOL_RETRIEVE_MULTI_HOP,
            TOOL_RETRIEVE_DOMAIN,
            TOOL_SUMMARIZE_DOCUMENT,
            TOOL_EXECUTIVE_SUMMARY,
            TOOL_COMPARE_DOCUMENTS,
            TOOL_CITE_SOURCES,
            TOOL_VERIFY_ANSWER,
            TOOL_REWRITE_QUERY,
        ]:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered MCP tool: {tool.name} ({tool.category})")

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool by name."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"Unregistered MCP tool: {tool_name}")
            return True
        return False

    def get(self, name: str) -> Optional[AgentTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[AgentTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_tools_by_category(self, category: str) -> List[AgentTool]:
        """List tools belonging to a specific category."""
        return [t for t in self._tools.values() if t.category == category]

    def list_categories(self) -> List[str]:
        """List all distinct tool categories."""
        return list(set(t.category for t in self._tools.values()))

    def list_tool_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def to_mcp_list_tools_result(self) -> List[Dict[str, Any]]:
        """Convert all tools to MCP ListToolsResult-compatible format."""
        return [t.to_mcp_dict() for t in self._tools.values()]

    def get_category_descriptions(self) -> Dict[str, str]:
        """Get a description for each tool category (for clustering)."""
        descriptions = {}
        for tool in self._tools.values():
            if tool.category not in descriptions:
                descriptions[tool.category] = tool.description
            else:
                descriptions[tool.category] += " " + tool.description
        return descriptions

    def get_tool_embeddings_text(self) -> Dict[str, str]:
        """
        Get embedding-friendly text for each tool.
        Used by K-Means to cluster tools by capability.
        """
        return {
            t.name: f"{t.category}: {t.description} Tags: {' '.join(t.domain_tags)}"
            for t in self._tools.values()
        }


# Singleton registry
agent_tool_registry = AgentToolRegistry()

