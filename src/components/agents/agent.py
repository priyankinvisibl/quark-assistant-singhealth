import time
from typing import Any

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage, Document

from src import utils
from src.components.agents.helpers import get_generator, get_generatorv2
from src.components.retrievers.pubmed import PubMedRetriever
from src.config.types import Config, Message, Prompt, Toolset

context_routes = [
    {
        "condition": "{{ not message['knowledgebase'] }}",
        "output": "{{message['content']}}",
        "output_name": "pubmed-retriever",
        "output_type": str,
    },
    {
        "condition": "{{ message['knowledgebase']|length > 0 }}",
        "output": "{{message['content']}}",
        "output_name": "context-retriever",
        "output_type": str,
    },
]


@component
class Deprecated:
    def __init__(
        self,
        config: Config,
        name: str = "default",
        multimodal: bool = False,
        image: bool = False,
    ):
        self._image = image
        self._functions = None
        self.toolset = None
        self.history = None
        self._document = None
        self.context_aware = False
        self.agent = name
        self.settings = config.agents.get(name)
        self.vendor = self.settings.vendor
        self.model, self._llm = get_generator(
            config.models[self.vendor], self.settings, multimodal, image
        )
        if multimodal or image:
            return
        template = Prompt.from_yaml_file(
            config.paths["prompts"].format(prompt=self.settings.prompt)
        ).messages

        # Google's API does not support model/system prompt as the first one
        if self.vendor == "google":
            template[0].role = "user"
            template.insert(1, ChatMessage.from_system(content="Understood"))
        self._pipe = Pipeline()
        self._pipe.add_component(f"{self.agent}_llm", self._llm)
        self._pipe.add_component(
            f"{self.agent}_prompt", ChatPromptBuilder(template=template)
        )
        self._pipe.connect(f"{self.agent}_prompt.prompt", f"{self.agent}_llm.messages")
        # self._pipe.draw(f"diagrams/{jpeg}.name")

    # NOTE: experimental
    @classmethod
    def with_attachments(
        cls,
        config: Config,
        document: dict[str, Any],
        name: str = "default",
    ):
        a = cls(config, name, multimodal=True)
        a._document = document
        # a._pipe.draw(f"diagrams/{name}.jpeg")
        return a

    @classmethod
    def with_tools(
        cls,
        config: Config,
        name: str = "default",
    ):
        a = cls(config, name)
        a.toolset = Toolset.from_yaml_file(
            config.paths["toolsets"].format(toolset=a.settings.toolset)
        )
        if a.vendor == "google":
            a._llm._tools = a.toolset.vertex_declarations()
        a._functions = {i.spec["function"]["name"]: i.function for i in a.toolset.tools}
        # a._pipe.draw(f"diagrams/{name}.jpeg")
        return a

    @classmethod
    def with_context(
        cls,
        config: Config,
        ret: "Retriever",  # undefined
        history: list[Message] = None,
        name: str = "default",
    ):
        a = cls(config, name)
        a.context_aware = True
        a.history = history
        retriever = ret
        a._pipe.add_component(
            "context-router", ConditionalRouter(routes=context_routes)
        )
        a._pipe.add_component("context-retriever", retriever)
        a._pipe.add_component("pubmed-retriever", PubMedRetriever())
        a._pipe.add_component("joiner", BranchJoiner(list[Document]))
        a._pipe.connect("context-router.pubmed-retriever", "pubmed-retriever")
        a._pipe.connect("context-router.context-retriever", "context-retriever.query")
        a._pipe.connect("pubmed-retriever.documents", "joiner")
        a._pipe.connect("context-retriever.documents", "joiner")
        a._pipe.connect("joiner", f"{name}_prompt.context")
        # print(a._pipe.dumps())
        # a._pipe.draw(f"diagrams/{name}.jpeg")
        return a

    # TODO: Cleanup and chang Dict[str, Any] -> Message
    @component.output_types(response=dict[str, Any])
    def run(self, message: dict[str, Any]):
        # TODO: integrated with prompt builder
        if self.toolset:
            result = self.sequential_tool_run(message)
        elif self._document:
            # response = self._llm.run(parts=[message["content"], *self._document["source"]["bytes"]]) #NOTE: Vertex
            response = self._llm.run(
                [message.as_chat_message()], document=self._document
            )  # NOTE: Bedrock
            message = message.to_dict()
            result = Message(
                origin=self.agent,
                role="assistant",
                content=response["replies"][0].content,
                metadata={"model": self.model},
                memory_id=message["memory_id"],
            )

        elif self._image:
            result = self._llm.run(prompt=message["content"])
        else:
            result = self.normal_run(message)
        return {"response": result.to_dict()}

    def normal_run(self, message: dict[str, Any]):
        context = None
        variables = {f"{self.agent}_prompt": {"message": message["content"]}}
        if self.context_aware:
            variables["context-router"] = {"message": message}
            if self.history:
                variables[f"{self.agent}_prompt"]["history"] = self.history
        response = self._pipe.run(
            variables,
            include_outputs_from=[
                f"{self.agent}_prompt",
                "joiner",
                "context-retriever",
                "context-router",
            ],
        )
        joiner = response.get("joiner", None)
        if joiner and joiner.get("value", []):
            print(message)
            source = list(response["context-router"].keys())[0]
            context = {
                "content": "\n".join([d.content for d in joiner["value"]]),
                "source": source,
            }
        return Message(
            origin=self.agent,
            role="assistant",
            content=response[f"{self.agent}_llm"]["replies"][0].content,
            metadata={"context": context, "model": self.model},
            memory_id=message["memory_id"],
            knowledgebase=message["knowledgebase"],
        )

    def sequential_tool_run(self, message: dict[str, Any]):
        messages = self._pipe._run_component(
            f"{self.agent}_prompt", {"message": message["content"]}
        )["prompt"][0]
        limit = 0
        while True:
            print(messages.content)
            res = self._pipe._run_component(
                f"{self.agent}_llm", {"messages": [messages]}
            )
            cm = res["replies"][0]
            if cm.name not in self._functions.keys() or limit >= 10:
                result = Message(
                    origin=self.agent,
                    role="assistant",
                    content=res["replies"][0].content,
                    metadata={"model": self.model},
                    memory_id=message["memory_id"],
                )
                return result
            function_name = cm.name
            function_response = self._functions[cm.name](**cm.content)
            messages.content += f"\nOutput of {function_name}: {function_response}\n"
            time.sleep(2)

    def to_dict():
        return self._pipe.to_dict()


def run_agent(
    message: Message,
    config: Config,
    name: str = "default",
) -> Message | None:
    settings = config.agents[name]
    vendor = settings.vendor
    model, llm = get_generatorv2(config.models[vendor], settings)
    prompt_config = utils.read_yaml(config.paths["prompts"].format(prompt=name))
    prompt = prompt_config["messages"][0]["content"]
        
    agent = Agent(chat_generator=llm, tools=[])
    if message.content is not None:
        resp = agent.run(
            messages=[ChatMessage.from_user(prompt.format(content=message.content))]
        )
        if not resp or "messages" not in resp or len(resp["messages"]) == 0:
            raise Exception("Did not receive expected response from the agent")
        return Message(
            content=resp["messages"][-1].content,
            origin=name,
            memory_id=message.memory_id,
            role="assistant",
            metadata=message.metadata,
        )
    return None


def run_pipeline(
    message: Message,
    config: Config,
    name: str = "default",
) -> Message | None:
    settings = config.agents[name]
    vendor = settings.vendor
    model, llm = get_generatorv2(config.models[vendor], settings)
    prompt_config = utils.read_yaml(config.paths["prompts"].format(prompt=name))
    prompt = prompt_config["messages"][0]["content"]
    
        
    _pipe = Pipeline()
    _pipe.add_component(f"{name}_llm", llm)
    _pipe.add_component(
        f"{name}_prompt",
        ChatPromptBuilder([ChatMessage.from_user(prompt)]),
    )
    _pipe.connect(f"{name}_prompt.prompt", f"{name}_llm.messages")
    if message.content is not None:
        resp = _pipe.run(data={"content": message.content}).get(f"{name}_llm", {})
        print(resp)
        if not resp or "replies" not in resp or len(resp["replies"]) == 0:
            raise Exception("Did not receive expected response from the agent")
        return Message(
            content=resp["replies"][-1].texts[-1],
            origin=name,
            memory_id=message.memory_id,
            role="assistant",
            metadata=message.metadata,
        )
    return None
