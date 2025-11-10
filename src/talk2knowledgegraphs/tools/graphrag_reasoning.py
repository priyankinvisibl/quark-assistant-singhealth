import logging
from textwrap import dedent

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Tool
from haystack.utils import Secret
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

from ...config.types import Config


class GraphRAGReasoningTool:
    """A tool that performs GraphRAG reasoning.

    Answer user queries using the KG as context. It works by using a textualized
    subgraph as context.
    """

    def __init__(
        self,
        history: str,
        document_store: InMemoryDocumentStore,
        model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Config | None = None,
    ):
        self.tool = Tool(
            name="graph_rag_reasoning",
            description=(
                "The tool for GraphRAG reasoning, i.e., answering user queries using "
                "the KG as context."
            ),
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            function=self._run,
        )
        self.retrieval_pipeline = self._get_retrieval_pipeline(
            document_store, model, embedding_model, config
        )
        self.history = history

    def _get_retrieval_pipeline(
        self,
        document_store: InMemoryDocumentStore,
        model: str,
        embedding_model: str,
        config: Config | None = None,
    ) -> Pipeline:
        """Get the retrieval pipeline for the tool."""
        template = [
            ChatMessage.from_system(
                dedent(
                    """
                    You are an expert at understanding a user's question, retrieving
                    relevant information store in the form of a knowledge graph, and
                    answering the question in natural language.

                    The knowledge graph will be given to you in the form of a JSON
                    object. The object contains an array of nodes, and an array of
                    edges. The edges array simply maps edges that start from one node
                    (from_index) and end at another (to_index). The nodes array simply
                    defines each node, along with their index (node_index), which should
                    help you map the data in the edges array.

                    You will also receive the schema for the graph. It will contain
                    metadata about the nodes and the edges that may not be present in
                    the graph itself.

                    Finally, you will receive a relevant selection of the history of the
                    conversation the user has been having. Use this to inform your
                    answer if context is needed.

                    Remember to respond only with your answer, and no preamble. If you
                    are not able to answer, say, "I don't know."
                    """
                ).strip()
            ),
            ChatMessage.from_user(
                dedent(
                    """
                    User's question: {{ prompt }}

                    Conversation history:
                    {{ history }}

                    Relevant Information:
                    {% for document in documents %}
                        {{ document.id }}:
                        {{ document.content }}
                    {% endfor %}
                    """
                ).strip()
            ),
        ]
        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component(
            "embedder", SentenceTransformersTextEmbedder(model=embedding_model)
        )
        retrieval_pipeline.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store=document_store)
        )
        retrieval_pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=template, required_variables=["prompt", "history", "documents"]
            ),
        )

        # Use cross-account permissions if needed.
        generator_kwargs = {"model": model, "generation_kwargs": {"temperature": 0.0}}
        if config is not None:
            cross_account_settings = (
                config.models.get("aws", {}).get("values", {}).get("crossAccount")
            )
            if cross_account_settings is not None:
                cross_account_id = cross_account_settings.get("id")
                cross_account_role = cross_account_settings.get("role")
                if cross_account_id is None or cross_account_role is None:
                    raise Exception("Cross-account set in config without ID or role")

                import boto3

                sts_connection = boto3.client("sts")
                cross_account = sts_connection.assume_role(
                    RoleArn=(
                        f"arn:aws:iam::{cross_account_id}:role/{cross_account_role}"
                    ),
                    RoleSessionName="quark-assistant",
                )
                generator_kwargs.update(
                    {
                        "aws_access_key_id": Secret.from_token(
                            cross_account["Credentials"]["AccessKeyId"]
                        ),
                        "aws_secret_access_key": Secret.from_token(
                            cross_account["Credentials"]["SecretAccessKey"]
                        ),
                        "aws_session_token": Secret.from_token(
                            cross_account["Credentials"]["SessionToken"]
                        ),
                    }
                )
        retrieval_pipeline.add_component(
            "llm", AmazonBedrockChatGenerator(**generator_kwargs)
        )

        retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        retrieval_pipeline.connect("retriever", "prompt_builder.documents")
        retrieval_pipeline.connect("prompt_builder.prompt", "llm.messages")
        return retrieval_pipeline

    def _run(self, prompt: str) -> str:
        """Run the GraphRAG reasoning tool.

        Parameters
        ----------
        prompt: str
            The prompt from the user.

        Returns
        -------
        response: str
            The text response from the LLM.
        """
        logging.debug(
            "Running GraphRAG tool; generating response from the retrieval pipeline..."
        )
        pipeline_response = self.retrieval_pipeline.run(
            {
                "embedder": {"text": prompt},
                "prompt_builder": {"prompt": prompt, "history": self.history},
            }
        )
        logging.debug("Retrieval complete")
        replies = pipeline_response.get("llm", {}).get("replies", [])
        if len(replies) > 0:
            return replies[0].text

        exc_msg = "Did not receive a response from the LLM"
        logging.exception(exc_msg)
        raise Exception(exc_msg)
