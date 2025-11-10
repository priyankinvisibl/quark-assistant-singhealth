import logging
from textwrap import dedent
from typing import Any

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack_integrations.components.generators.amazon_bedrock import (
    AmazonBedrockChatGenerator,
)

from src.components.agents.helpers import get_boto3session_haystack

from ...config.types import Config


class Talk2KnowledgeGraphsAgent:
    """An agent that connects with a knowledge graph based on natural language input."""

    def __init__(
        self,
        history: str,
        model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        tools: list[Tool] | None = None,
        config: Config | None = None,
    ):
        if tools is None:
            tools = []

        exit_condition = ""
        for tool in tools:
            if tool.name == "graph_visualization":
                exit_condition = tool.name
            if tool.name == "graph_rag_reasoning" and not exit_condition:
                exit_condition = tool.name

        exit_conditions = []
        if exit_condition:
            exit_conditions = [exit_condition]

        # Use cross-account permissions if needed.
        generator_kwargs = {"model": model, "generation_kwargs": {"temperature": 0.0}}
        if config is not None:
            generator_kwargs.update(
                get_boto3session_haystack(config.models.get("aws", {}))
            )

        self.history = history
        self.agent = Agent(
            chat_generator=AmazonBedrockChatGenerator(**generator_kwargs),
            tools=tools,
            exit_conditions=exit_conditions,
        )
        self.agent.warm_up()

    def run(
        self, prompt: str, system_instructions: str | None = None
    ) -> dict[str, Any]:
        """Give the agent a prompt and run it."""
        # The following method checks if the agent is already warmed up first, so we
        # don't have to.
        self.agent.warm_up()

        history_enhanced_prompt = dedent(
            f"""
            User's question: {prompt}

            Conversational history:
            {self.history}
            """
        ).strip()
        messages = [ChatMessage.from_user(history_enhanced_prompt)]
        if system_instructions is not None:
            messages = [ChatMessage.from_system(system_instructions)] + messages
        logging.debug("Running the agent...")
        return self.agent.run(messages=messages)
