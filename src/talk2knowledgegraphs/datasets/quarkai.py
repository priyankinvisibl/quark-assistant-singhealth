import json
import logging
from ast import literal_eval
from typing import Any

import pandas as pd
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ..schemas import Edge, EdgeDirection, Node


class QuarkAIDataset:
    """Load the QuarkAI data structure into that required by the T2KG tool."""

    def __init__(self):
        self.name = "QuarkAI"
        self.schema_labels_to_df_columns = {
            "Gene": "geneName",
            "BiologicalProcess": "molecularFunctions",
            "CellularComponent": "cellularComponents",
            "Drug": "drugs",
            "Pathway": "pathways",
        }
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.document_store = InMemoryDocumentStore()

    def _load_edge_directionality(
        self, schema: dict[str, Any]
    ) -> dict[str, dict[str, EdgeDirection]]:
        """Load the edge directionality into a mapper based on node types.

        The directionality is stored in the schema. Edges may be in-edges, out-edges, or
        bi-directional.

        Parameters
        ----------
        schema: dict[str, Any]
            The schema for the knowledge graph.

        Returns
        -------
        directionality: dict[str, dict[str, EdgeDirection]]
            The mapping from one node type, to another node type, to the directionality
            of the edge between them.
        """
        directionality: dict[str, dict[str, EdgeDirection]] = {}

        logging.info("Getting edge directionality based on the schema...")
        for edge in schema.get("edges", []):
            from_node = self.schema_labels_to_df_columns.get(
                edge.get("from", {}).get("vertex_label", "")
            )
            to_node = self.schema_labels_to_df_columns.get(
                edge.get("to", {}).get("vertex_label", "")
            )
            if from_node is None or to_node is None:
                continue
            if from_node not in directionality:
                directionality[from_node] = {}
            if to_node not in directionality:
                directionality[to_node] = {}

            # Set the out-edge.
            if (
                to_node in directionality.get(from_node, {})
                and directionality.get(from_node, {}).get(to_node) != EdgeDirection.OUT
            ):
                directionality[from_node][to_node] = EdgeDirection.BI
            else:
                directionality[from_node][to_node] = EdgeDirection.OUT

            # Set the in-edge.
            if (
                from_node in directionality.get(to_node, {})
                and directionality.get(to_node, {}).get(from_node) != EdgeDirection.IN
            ):
                directionality[to_node][from_node] = EdgeDirection.BI
            else:
                directionality[to_node][from_node] = EdgeDirection.IN

        return directionality

    def _index_schema_and_graph(
        self,
        schema: dict[str, Any],
        sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Index the schema and graph into an in-memory document store."""
        documents = [Document(id="schema", content=json.dumps(schema))]
        if len(self.nodes) > 0:
            documents.append(
                Document(
                    id="nodes",
                    content=json.dumps([node.model_dump() for node in self.nodes]),
                )
            )
        if len(self.edges) > 0:
            documents.append(
                Document(
                    id="edges",
                    content=json.dumps([edge.model_dump() for edge in self.edges]),
                )
            )

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            instance=SentenceTransformersDocumentEmbedder(
                model=sentence_transformer_model
            ),
            name="doc_embedder",
        )
        indexing_pipeline.add_component(
            instance=DocumentWriter(document_store=self.document_store),
            name="doc_writer",
        )
        indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

        logging.info(
            "Indexing documents for %s...",
            ", ".join([document.id for document in documents]),
        )
        indexing_pipeline.run({"doc_embedder": {"documents": documents}})

    def load_data(self, schema: dict[str, Any], graph_df: pd.DataFrame) -> None:
        """Load the schema and the graph, and convert into nodes and edges.

        The graph is in the Quark representation. In this representation, columns
        represent nodes, and row represent edges from or two gene nodes. The
        directionality of each edge is defined in the schema based on the nodes the edge
        is between.
        """
        edge_directionality = self._load_edge_directionality(schema)
        # Use mappings to prevent duplicates, and refactor the data into the proper
        # structure at the end.
        edge_data: dict[int, dict[int, EdgeDirection]] = {}
        node_data: dict[str, Node] = {}
        node_idx = 0

        logging.info("Loading graph data...")
        for _, row in graph_df.iterrows():
            # Since each row corresponds to a gene, first set that node.
            gene_name = row.at["geneName"]
            if gene_name in node_data:
                continue
            node_data[gene_name] = Node(
                node_index=node_idx,
                node_name=gene_name,
                node_source="QuarkAI",
                node_id=gene_name,
                node_type="geneName",
            )
            gene_node_idx = node_idx
            node_idx += 1
            if gene_node_idx not in edge_data:
                edge_data[gene_node_idx] = {}

            for column, nodes in row.items():
                if (
                    column not in ["", "geneName"]
                    and "details" not in str(column).lower()
                    and isinstance(nodes, str)
                    and nodes.startswith("[")
                    and nodes.endswith("]")
                ):
                    nodes_as_obj = literal_eval(nodes)
                    for node in nodes_as_obj:
                        # Then, add other nodes if they aren't already in the mapping.
                        if node not in node_data:
                            node_data[node] = Node(
                                node_index=node_idx,
                                node_name=node,
                                node_source="QuarkAI",
                                node_id=node,
                                node_type=str(column),
                            )
                            node_idx += 1

                        # Finally, add an edge between the two.
                        direction = edge_directionality.get("geneName", {}).get(
                            str(column)
                        )
                        if direction is None:
                            raise Exception(
                                "Edge directionality from %s to %s not found in schema",
                                "geneName",
                                str(column),
                            )
                        edge_data[gene_node_idx][node_data[node].node_index] = direction
                if (
                    "details" in str(column).lower()
                    and isinstance(nodes, str)
                    and nodes.startswith("[")
                    and nodes.endswith("]")
                ):
                    nodes_as_obj = literal_eval(nodes)
                    for entity in nodes_as_obj:
                        entity_id = entity.get("entityId")
                        entity_name = entity.get("entityName", "")
                        if entity_id is not None and entity_id in node_data:
                            node_data[entity_id].node_name = entity_name

        self.nodes = list(node_data.values())

        edges = []
        for node0, node1_data in edge_data.items():
            for node1, direction in node1_data.items():
                if direction == EdgeDirection.IN:
                    edges.append(Edge(from_index=node1, to_index=node0))
                elif direction == EdgeDirection.OUT:
                    edges.append(Edge(from_index=node0, to_index=node1))
                elif direction == EdgeDirection.BI:
                    edges.append(Edge(from_index=node0, to_index=node1))
                    edges.append(Edge(from_index=node1, to_index=node0))
        self.edges = edges

        self._index_schema_and_graph(schema)

        logging.info("Data loading complete")
