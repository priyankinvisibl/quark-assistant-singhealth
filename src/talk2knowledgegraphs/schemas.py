from enum import Enum

from pydantic import BaseModel


class EdgeDirection(str, Enum):
    """The directions an edge can be."""

    BI = "bi"  # bidirectional edge
    IN = "in"
    OUT = "out"


class Edge(BaseModel):
    """The representation of a knowledge graph edge for Talk2KnowledgeGraph."""

    from_index: int
    to_index: int


class Node(BaseModel):
    """The representation of a knowledge graph node for Talk2KnowledgeGraph."""

    node_index: int
    node_name: str
    node_source: str
    node_id: str
    node_type: str


class Graph(BaseModel):
    """The representation of a knowledge graph or sub-graph for Talk2KnowledgeGraph."""

    nodes: list[Node]
    edges: list[Edge]
