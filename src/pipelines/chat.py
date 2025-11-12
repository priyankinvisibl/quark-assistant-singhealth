import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi.exceptions import HTTPException
from haystack import Pipeline
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ByteStream

from src.components.agents.agent import Deprecated, run_pipeline
from src.config.types import Config, Memory, Message, StorageConfig
from src.gtex.pipeline import QueryEnhancerPipeline
from src.internal.storage.s3.client import QClient
from src.internal.storage.types import KnowledgeStoreClient, MemoryClient
from src.talk2knowledgegraphs.agents import Talk2KnowledgeGraphsAgent
from src.talk2knowledgegraphs.datasets import QuarkAIDataset
from src.talk2knowledgegraphs.tools import GraphRAGReasoningTool
from src.utils import time_greater_than


# TODO: revamp
class Chat:
    def __init__(self, config: Config, mem_client=None, ks_client=None):
        self.config = config
        self.mem_client = mem_client
        self.ks_client = ks_client
        self.kg_client: QClient | None = None

    # TODO
    @staticmethod
    def build_pipes(self, config):
        pass

    # TODO:
    # link messages with prompt builder
    def summarize(
        self,
        memory_id: str = "",
        history: list[Message] = None,
        memory: Memory = None,
        count: int = 10,
        force: bool = False,
    ):
        if memory_id != "":
            history, memory = self.mem_client.get_history(memory_id)

        if force or (
            (len(history) > count)
            and (
                not memory.metadata.get("summary", None)
                or time_greater_than(history[1].timestamp, history[0].timestamp)
            )
        ):
            summarizer = Deprecated(self.config, "summarizer")
            result = summarizer._pipe.run(
                {"messages": history},
                include_outputs_from=["summarizer_prompt", "summarizer_prompt"],
            )
            memory.metadata["summary"] = Message(
                result["summarizer_llm"]["replies"][0].content, "assistant", None
            ).to_dict(
                exclude=(
                    "memory_id",
                    "message_id",
                    "metadata",
                    "knowledgebase",
                    "role",
                    "origin",
                )
            )
        self.mem_client.update_memory(memory_id, memory)
        return memory.metadata["summary"]

    # TODO:
    # Save routes to yaml
    # Create one documentstore index
    # Consolidate output through pipeline
    # optimize context flow
    def chat(self, message: Message):
        history, memory = self.mem_client.get_history(message.memory_id)
        embedding_ids = []
        if memory:
            message.knowledgebase = memory.knowledgebase
            if message.knowledgebase:
                kb = self.ks_client.get_knowledgebase(message.knowledgebase)
                embedding_ids = kb.get_embedding_ids()
        gen_ret = self.ks_client.get_retriever("bm25", embedding_ids)
        qa_ret = self.ks_client.get_retriever("bm25", embedding_ids)
        taskdecider_agent = Deprecated(self.config, "task-handler-v0")
        textgen_agent = Deprecated.with_context(
            self.config, gen_ret, history, name="text-gen"
        )
        reddi_agent = Deprecated(self.config, "re-ddi")
        rebc5cdr_agent = Deprecated(self.config, "re-bc5cdr")
        qa_agent = Deprecated.with_context(self.config, qa_ret, history, name="qa")
        # ncbi_agent = Agent.with_tools(self.config, "ncbi")
        # image_gen_agent = Agent.with_context(self.config, "image-gen")

        pipe = Pipeline()
        # pipe.add_component("task-node", BranchJoiner(Dict[str, Any]))
        pipe.add_component("task-handler", taskdecider_agent)
        pipe.add_component("task-router", ConditionalRouter(routes=task_routes))
        pipe.add_component("text-gen", textgen_agent)
        pipe.add_component("re-ddi", reddi_agent)
        pipe.add_component("re-bc5cdr", rebc5cdr_agent)
        pipe.add_component("qa", qa_agent)
        # pipe.add_component("ncbi", ncbi_agent)

        # task decider loop
        # pipe.connect("task-node", "task-handler")
        pipe.connect("task-handler.response", "task-router.response")
        # pipe.connect("task-router.task-node", "task-node")

        pipe.connect("task-router.qa", "qa.message")
        # pipe.connect("task-router.ncbi", "ncbi.message")
        pipe.connect("task-router.text-gen", "text-gen.message")
        pipe.connect("task-router.re-ddi", "re-ddi.message")
        pipe.connect("task-router.re-bc5cdr", "re-bc5cdr.message")
        # pipe.connect("task-router.image-gen", "image-gen.message")
        # pipe.draw(f"diagrams/pipe.jpeg")
        dict_message = message.to_dict()
        result = pipe.run(
            {
                "task-handler": {"message": dict_message},
                "task-router": {"message": dict_message},
            },
        )

        # {"task-node": {"value": dict_message}},
        response = Message(**result.pop(list(result.keys())[0])["response"])
        self.mem_client.add_messages(message)
        self.mem_client.add_messages(response)
        if not memory:
            memory = Memory(user=message.origin, memory_id=message.memory_id)
            self.mem_client.create_memory(memory)
            nomenclator = Deprecated(self.config, "nomenclator-title")
            result = nomenclator._pipe.run(
                {"content": message.content},
            )
            memory.name = result["nomenclator-title_llm"]["replies"][
                0
            ].content.removeprefix("Title: ")
            self.mem_client.update_memory(memory.memory_id, memory)
        # take only the first sumamry + 8 records and append the new message-response pair
        # self.summarize_memory(history=history[:8] + message_pair, memory)
        return response

    def gtex_chat(
        self,
        prompt: Message,
        schema_path: str,
        entities_path: str,
        entities_file_type: Literal["csv", "txt"],
        model_path: str,
    ) -> Message:
        """Generate a chat response using GTEx integration.

        This method integrates the GTEx pipeline to process the prompt and generate a
        response.

        Parameters
        ----------
        prompt: Message
            The user-provided prompt message.
        schema_path: str
            Path to the schema JSON file.
        entities_path: str
            Path to the directory containing files for entity mapping.
        entities_file_type: Literal["csv", "txt"]
            Whether the entities are in the CSV format or the TXT format.
        model_path: str
            Path to the directory that contains the NER model (hint: this directory
            should contain the `config.cfg` file).

        Returns
        -------
        response: Message
            The assistant's response.
        """
        prompt_content = prompt.content
        if prompt_content is None:
            raise HTTPException(status_code=422, detail="content is a mandatory field")

        # TODO: use history
        # history, memory = self.mem_client.get_history(prompt.memory_id)
        # Initialize the QueryEnhancerPipeline.
        schema = {}
        try:
            with open(schema_path) as f:
                if schema_path.endswith(".json"):
                    import json

                    schema = json.load(f)
                elif schema_path.endswith(".yaml"):
                    import yaml

                    schema = yaml.safe_load(f)
                else:
                    raise Exception("Schema must be a JSON or YAML file")
        except Exception as e:
            raise Exception(f"Failed to load schema or graph data: {e}") from e

        pipeline = QueryEnhancerPipeline(schema=schema, config=self.config)

        # Run the pipeline.
        pipeline_response = pipeline.run(
            prompt=prompt_content,
            model_path=Path(model_path),
            entities_path=Path(entities_path),
            entity_file_type=entities_file_type,
            llm=self.config.models.get("aws", {}).get("name"),
        )

        # Create the response message.
        response = Message(
            content=pipeline_response.nl_db_response,
            origin=prompt.origin,
            memory_id=prompt.memory_id,
            role="assistant",
        )

        # self.mem_client.add_messages(prompt)
        # self.mem_client.add_messages(response)
        # if not memory:
        #     memory = Memory(user=prompt.origin, memory_id=prompt.memory_id)
        #     self.mem_client.create_memory(memory)
        #     result = run_pipeline(prompt, self.config, "nomenclator-title")
        #     if result is None or result.content is None:
        #         raise HTTPException(
        #             status_code=500, detail="error generating name for conversation"
        #         )
        #     memory.name = result.content.removeprefix("Title: ")
        #     self.mem_client.update_memory(memory.memory_id, memory)
        return response

    def knowledge_graph_chat(
        self, prompt: Message, graph_location: str, schema_location: str
    ) -> Message:
        """Generate a chat response from the knowledge graph–backed agent.

        The chat follows this flow.

        1. Add the prompt message to the memory.
        2. Load the schema and graph data.
        3. Initialize the agent with the reasoning tool.
        4. Generate a response using the agent.
        5. Convert the agent's response into a ``Message`` object.
        6. Add the response message to the memory.
        7. Return the response message.

        Parameters
        ----------
        message: Message
            The prompt.
        graph_location: str
            Path to the graph CSV file.
        schema_location: str
            Path to the schema JSON file.

        Returns
        -------
        response: Message
            The assistant's response.

        Raises
        ------
        Exception
            If the agent does not return a response.
        """
        prompt_content = prompt.content
        if prompt_content is None:
            raise HTTPException(status_code=422, detail="content is a mandatory field")

        # Get the graph from the object-store.
        if self.kg_client is None:
            self.kg_client = QClient(
                config=StorageConfig(enabled=True, config={}, stores={}),
            )
        graph_file_name = graph_location.split("/")[-1]

        # Create a temporary path and store the file there.
        tmp_graph_path = "./tmp/objectstore/"
        if not os.path.exists(tmp_graph_path):
            os.makedirs(tmp_graph_path, exist_ok=True)
        tmp_graph_location = f"{tmp_graph_path}{graph_file_name}"
        graph_obj = self.kg_client.get_record(
            graph_location, graph_location.split("/")[-1]
        )
        open_mode = "w"
        if isinstance(graph_obj, bytes):
            open_mode = "wb"
        with open(tmp_graph_location, open_mode) as f:
            f.write(graph_obj)

        schema = {}
        try:
            with open(schema_location) as f:
                if schema_location.endswith(".json"):
                    import json

                    schema = json.load(f)
                elif schema_location.endswith(".yaml"):
                    import yaml

                    schema = yaml.safe_load(f)
                else:
                    raise Exception("Schema must be a JSON or YAML file")
            graph_df = pd.read_csv(tmp_graph_location)
        except Exception as e:
            os.remove(tmp_graph_location)
            raise Exception(f"Failed to load schema or graph data: {e}") from e

        # Prepare the dataset.
        dataset = QuarkAIDataset()
        dataset.load_data(schema, graph_df)

        # history, memory = self.mem_client.get_history(prompt.memory_id)
        # if memory is None:
        #     raise HTTPException(
        #         status_code=404, detail=f"Memory {prompt.memory_id} not found"
        #     )
        # if history is None:
        #     history_context = "(No relevant conversation history!)"
        # else:
        #     history_context = "\n\n".join(
        #         f"{message.role}: {message.content}" for message in history
        #     )

        # Initialize the reasoning tool.
        reasoning_tool = GraphRAGReasoningTool(
            model=self.config.models.get("aws", {}).get("name"),
            history="(No relevant conversation history!)",  # history_context,
            document_store=dataset.document_store,
            config=self.config,
        )

        # Initialize the agent and run it.
        agent = Talk2KnowledgeGraphsAgent(
            model=self.config.models.get("aws", {}).get("name"),
            history=[],  # history,
            tools=[reasoning_tool.tool],
            config=self.config,
        )
        agent_response = agent.run(prompt_content)

        # Process the agent's response
        if (
            not agent_response
            or "messages" not in agent_response
            or len(agent_response["messages"]) == 0
        ):
            os.remove(tmp_graph_location)
            raise Exception("Did not receive a response from the knowledge graph agent")

        final_message = agent_response["messages"][-1]
        content: str | None = None
        if final_message.tool_call_result is not None:
            content = final_message.tool_call_result.result
        else:
            content = final_message.text
        if content is None or not content:
            os.remove(tmp_graph_location)
            raise Exception(
                "Did not receive a valid response from the knowledge graph agent"
            )

        response = Message(
            content=content,
            origin=prompt.origin,
            memory_id=prompt.memory_id,
            role="assistant",
            metadata=prompt.metadata,
        )
        # self.mem_client.add_messages(prompt)
        # self.mem_client.add_messages(response)
        os.remove(tmp_graph_location)
        return response

    # TODO: all methods from here are EXPERIMENTAL
    def multimodal_chat(
        self,
        message: Message,
        f: tuple[str, bytes, str],
        # attachments: List[str] | None = None,
    ):
        # filetype = f[0]
        # _bytes = [ByteStream(data=f[1], mime_type=f[0])]
        # _bytes = [ByteStream(data=f[1], mime_type=f[0])]
        # TODO: integrate with url router in filecontroller
        # elif attachments:
        #     _bytes = [
        #         ByteStream(data=requests.get(url).content, mime_type=mime_type)
        #         for url in attachments
        #     ]

        multimodal_agent = Deprecated.with_attachments(
            self.config,
            name="default",
            document={
                "document": {
                    "name": "document",
                    "format": "pdf",
                    "source": {"bytes": f[1]},
                }
            },
        )
        response = multimodal_agent.run(message)
        response = Message(**response["response"])
        response.metadata["reference"] = f[2]
        print(response)
        self.mem_client.add_messages(message)
        self.mem_client.add_messages(response)
        return response

    def multimodal_image(
        self,
        message: Message,
        f: tuple[str, bytes] = None,
        attachments: list[str] = None,
    ):
        if f:
            _bytes = [ByteStream(data=f[1], mime_type=f[0])]
        # TODO: integrate with url router in filecontroller
        elif attachments:
            _bytes = [
                ByteStream(data=requests.get(url).content, mime_type=mime_type)
                for url in attachments
            ]
        multimodal_agent = Deprecated.with_attachments(
            self.config, name="default", filebytes=_bytes
        )
        response = multimodal_agent.run(message.to_dict())
        # response = multimodal_agent.run(message.to_dict())
        image_agent = Deprecated(self.config, name="image-gen", image=True)
        print(response["response"]["content"])
        result = image_agent._llm.run(prompt=response["response"]["content"])
        return result["images"][0].data

    def code(self, input_: str):
        developer = Deprecated(self.config, "developer")

        response = {
            "fileContent": developer.run({"input": input_}),
            # "fileContent": p.run({"input": input_})["developer"]["response"][0].rstrip(),
        }
        return response


task_routes = [
    {
        "condition": "{{response['content'].lstrip('Task: ') == '1' }}",
        "output": "{{message}}",
        "output_name": "text-gen",
        "output_type": dict[str, any],
    },
    {
        "condition": "{{response['content'].lstrip('Task: ') == '2' }}",
        "output": "{{message}}",
        "output_name": "re-bc5cdr",
        "output_type": dict[str, any],
    },
    {
        "condition": "{{response['content'].lstrip('Task: ') == '3' }}",
        "output": "{{message}}",
        "output_name": "re-ddi",
        "output_type": dict[str, any],
    },
    {
        "condition": "{{response['content'].lstrip('Task: ') == '4' }}",
        "output": "{{message}}",
        "output_name": "qa",
        "output_type": dict[str, any],
    },
    # {
    #     "condition": "{{response['content'].lstrip('Task: ') == '5' }}",
    #     "output": "{{message}}",
    #     "output_name": "ncbi",
    #     "output_type": dict[str, any],
    # },
    {
        "condition": "{{response['content'].lstrip('Task: ') == '6' }}",
        "output": "{{message}}",
        "output_name": "image-gen",
        "output_type": dict[str, any],
    },
    {
        "condition": "{{response['content'].lstrip('Task: ') == '-1' }}",
        "output": "{{message}}",
        "output_name": "qa",  # TODO: default is qa
        "output_type": dict[str, any],
    },
]
# {
#     "condition": "{{response['content'].lstrip('Task: ') == '-1' }}",
#     "output": "{{message}}",
#     "output_name": "task-node",
#     "output_type": Dict[str, Any],
# },
