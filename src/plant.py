import importlib
from typing import Any

import boto3
from haystack import Pipeline
from haystack.core.component import Component
from haystack.dataclasses import ChatMessage

from src.config.state import State
from src.internal.storage.opensearch.knowledgestore import KSClient
from src.internal.storage.opensearch.memory import MemClient
from src.internal.storage.opensearch.promptstore import QClient
from src.internal.storage.types import (
    KnowledgeStoreClient,
    Memory,
    MemoryClient,
    Message,
    PromptStoreClient,
)
from src.pipelines import chat, embed
from src.utils import get_boto3_creds

# TODO: tmp
llms = {
    "vertex": "haystack_integrations.components.generators.google_vertex.VertexAIGeminiGenerator",
    "bedrock": "haystack_integrations.components.generators.amazon_bedrock.AmazonBedrockChatGenerator",
}


class Plant:
    def __init__(self, state: State):
        self.state = state

    # def get_knowledgebase_pipelines(self) -> memorypipelines:
    #     return KnowledgebasePipeline(state)

    # TODO:
    # make use of memstore and knowledgebase more dynamic
    # revamp pipes
    def get_chat_client(self) -> chat.Chat:
        kb_storage = self.state.get_storage("opensearch")
        mem_storage = self.state.get_storage("opensearch")
        kb_client = KSClient(kb_storage)
        mem_client = MemClient(mem_storage)
        config = self.state.config
        return chat.Chat(config, mem_client, kb_client)

    def get_embed_client(self) -> embed.Embed:
        kb_storage = self.state.get_storage("opensearch")
        kb_client = KSClient(kb_storage)
        s3_storage = self.state.get_storage("s3")
        return embed.Embed(kb_client, s3_storage)

    def get_knowledgestore_client(self) -> KnowledgeStoreClient:
        return KSClient(self.state.get_storage("opensearch"))

    def get_memory_client(self) -> MemoryClient:
        return MemClient(self.state.get_storage("opensearch"))

    def get_promptstore_client(self) -> PromptStoreClient:
        return QClient(self.state.get_storage("opensearch"))

    def get_chat_generator(self, provider, config: dict[str, Any]):
        module, cls = llms[provider].rsplit(".", 1)
        mod = importlib.import_module(module)
        return getattr(mod, cls)(**config)

    def run_components(
        self,
        components: dict[str, Component],
        values: dict[str, Any],
        message: dict[str, Any],
    ) -> dict[str, Any]:
        pipe = Pipeline()
        for name, c in components.items():
            pipe.add_component(name, c)
        values["messages"] = [ChatMessage.from_user(message["content"])]
        return pipe.run(values)

    # TODO: move
    def run_agent(self, message: Message, memory: Memory):
        # TODO: documents are not supported by haystack chatmessages as of yet

        system_prompts = []
        bedrock_input = {
            "role": "user",
            "content": [{"text": message.content}],
        }
        agent = message.metadata.get("agent")
        import yaml

        with open(self.state.config.paths["prompts"].format(prompt=agent)) as f:
            prompt_yaml = yaml.safe_load(f)
            prompt = prompt_yaml["messages"][0]["content"]
            text_input = prompt.format(**message.metadata.get("params", {}))
            system_prompts = [{"text": text_input}]
            # TODO:
            if bedrock_input["content"][0]["text"] is None:
                bedrock_input["content"][0]["text"] = text_input
            
        # Get temperature from prompt config if available
        temperature = prompt_yaml.get("temperature", None)

        if s3path := message.metadata.get("document", ""):
            from pathlib import Path

            p = Path(s3path)
            bedrock_input["content"].append(
                {
                    "document": {
                        "name": "document",
                        "format": p.suffix.removeprefix("."),
                        "source": {
                            "bytes": self.state.get_storage("s3").get_record(
                                s3path, p.name
                            )
                        },
                    }
                }
            )

        model = self.state.config.models.get("aws", {}).get(
            "name", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        session = boto3.Session(
            **get_boto3_creds(self.state.config.models.get("aws", {}))
        )
        bedrock = session.client(service_name="bedrock-runtime")
        bedrock_response = bedrock.converse(
            modelId=model,
            messages=[bedrock_input],
            system=system_prompts,
            inferenceConfig={"temperature": temperature} if temperature is not None else {},
        )
        response = Message(
            origin=message.metadata["agent"],
            role="assistant",
            content=bedrock_response["output"]["message"]["content"][0]["text"],
            metadata={
                "context": message.metadata.get("document", ""),
                "model": model,
                # "hidden": True,
            },
            memory_id=message.memory_id,
        )
        # TODO: error in creating objects?
        self.get_memory_client().create_memory(memory)
        self.get_memory_client().add_messages(message)
        self.get_memory_client().add_messages(response)
        # else:
        #     from src.components.agents.agent import DepecreatedAgent

        #     ncbi_agent = DepecreatedAgent.with_tools(self.state.config, "ncbi")
        #     message = Message(
        #         origin="quark-analytics",
        #         role="assistant",
        #         content=parameters["input"],
        #         memory_id="analytics",
        #     )
        #     out = ncbi_agent.run(message.to_dict())
        #     print(out)
        return response
