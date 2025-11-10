import importlib
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from dataclass_wizard import JSONWizard, YAMLWizard
from datargs import arg, argsclass
from haystack.components.builders.chat_prompt_builder import (
    ChatPromptBuilder,
)
from haystack.dataclasses.chat_message import ChatMessage
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from vertexai.preview.generative_models import FunctionDeclaration, Tool

import src.utils as utils


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class ResponseModel(BaseSchema):
    title: str
    data: Any
    metrics: dict[str, Any]
    collections: dict[str, Any]


class ResponseCollection(BaseSchema):
    title: str
    items: list[Any] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    collections: dict[str, Any] = Field(default_factory=dict)


# TODO: change to alias
class PromptMetadata(BaseSchema):
    name: str
    collection: str
    version: str
    description: str | None = None
    project: str | None = None
    owner: str | None = None
    createdAt: str = Field(default_factory=utils.get_zulu_time)
    updatedAt: str = Field(default_factory=utils.get_zulu_time)
    id: str | None = None
    tags: dict[str, str] | None = None
    sharedWith: list[str] | None = None

    @classmethod
    def from_api(
        cls,
        name: str,
        collection: str,
        description: str,
        project: str,
        user: str,
        version: str,
        id_,
    ):
        return cls(
            name=name,
            collection=collection,
            description=description,
            project=project,
            owner=user,
            version=version,
            id=id_,
        )


class PromptRecord(BaseSchema):
    role: str
    content: str
    order: int
    name: str | None = None

    def as_chatmessage(self) -> ChatMessage:
        return ChatMessage.from_dict(self.model_dump(exclude={"order", "cmessage"}))


class PromptVariable(BaseSchema):
    name: str
    data_type: str = Field(default="str")
    description: str | None = None
    default_value: str | None = None
    optional: bool = True


class PromptModel(BaseSchema):
    metadata: PromptMetadata
    prompts: list[PromptRecord]
    variables: list[PromptVariable] | None = None

    @classmethod
    def from_api(
        cls,
        name: str,
        collection: str,
        description: str | None,
        project: str,
        user: str,
        version: str,
        prompts: list[PromptRecord],
        variables: list[PromptVariable] | None = None,
    ):
        meta = PromptMetadata(
            name=name,
            collection=collection,
            description=description,
            project=project,
            owner=user,
            version=version,
            id=f"prompts_{uuid.uuid4()}",
        )
        return cls(metadata=meta, variables=variables, prompts=prompts)

    def model_post_init(self, __context) -> None:
        self.prompts = sorted(
            self.prompts, key=lambda prompt: prompt.order, reverse=False
        )

    def get_builder(self):
        return ChatPromptBuilder(
            template=self.get_prompt_template(),
            variables=self.get_variable_list(),
            required_variables=self.get_mandatory_variables(),
        )

    def get_mandatory_variables(self) -> list[str]:
        names = []
        if self.variables:
            names = [
                variable.name for variable in self.variables if not variable.optional
            ]
        return names

    def get_variable_list(self) -> list[str]:
        names = []
        if self.variables:
            names = [variable.name for variable in self.variables]
        return names

    def get_prompt_template(self) -> list[ChatMessage]:
        return [prompt.as_chatmessage() for prompt in self.prompts]


@dataclass(frozen=True, slots=True)
class User:
    email: str
    name: str


@dataclass(slots=True)
class Embedding:
    index_id: str
    filename: str
    filepath: str
    embedder: str
    embedding_properties: dict[str, Any] = field(default_factory=dict)
    embedded_at: str = field(default_factory=lambda: utils.get_zulu_time())


@dataclass(slots=True)
class KnowledgeBase(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    id: str
    project: str
    owner: str
    name: str
    version: str
    embeddings: list[Embedding] = field(default_factory=list)
    shared_with: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: utils.get_zulu_time())
    updated_at: str = field(default_factory=lambda: utils.get_zulu_time())

    @classmethod
    def new(cls, user: str, project: str, name: str):
        return KnowledgeBase(f"kb_{uuid.uuid4()}", project, user, name, "v1")

    def get_embedding_ids(self):
        return [embedding.index_id for embedding in self.embeddings]


@dataclass(slots=True)
class ToolItem:
    spec: dict[str, Any]
    module: str
    tests: list[dict[str, Any]] | None = None
    function: callable = field(init=False)
    vertex_declaration: FunctionDeclaration = field(init=False)

    def __post_init__(self):
        # TODO: validate how safe this is
        mod = importlib.import_module(f"src.tools.{self.module}")
        self.function = getattr(mod, self.spec["function"]["name"])
        self.vertex_declaration = FunctionDeclaration(
            name=self.spec["function"]["name"],
            description=self.spec["function"]["description"],
            parameters=self.spec["function"]["parameters"],
        )

    def ping(self):
        return all(
            self.function(**items["input"]) == items["output"] for items in self.tests
        )


@dataclass(slots=True)
class Toolset(YAMLWizard):
    version: str
    name: str
    description: str
    tools: list[ToolItem]

    def vertex_declarations(self):
        funcs = [tool.vertex_declaration for tool in self.tools]
        return [Tool(funcs)]


# TODO: include knowledgebase with version
@dataclass(slots=True)
class Memory(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    memory_id: str
    user: str
    knowledgebase: str = ""
    name: str | None = None
    created_at: str = field(default_factory=lambda: utils.get_zulu_time())
    updated_at: str = field(default_factory=lambda: utils.get_zulu_time())
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, user: str):
        return Memory(f"mem_{uuid.uuid4()}", user)

    def __post_init__(self):
        try:
            _ = self.metadata["archive"]
        except KeyError:
            self.metadata["archive"] = False


# TODO: combine ChatMessage and Message
@dataclass(slots=True)
class Message(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    origin: str
    memory_id: str
    content: str | None = None
    knowledgebase: str | None = None
    role: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: utils.get_zulu_time())
    message_id: str = field(default_factory=lambda: f"message_{uuid.uuid4()}")

    # def as_chat_message(self):
    #     return ChatMessage.from_dict(
    #         {"content": self.content, "role": self.role, "name": None}
    #     )


@dataclass(slots=True)
class Prompt(YAMLWizard):
    version: str
    name: str
    description: str
    messages: list[ChatMessage]


@dataclass(frozen=True, slots=True)
class AgentSettings:
    vendor: str
    toolset: str | None = None
    prompt: str | None = None
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StorageConfig:
    enabled: bool
    config: dict[str, Any]
    stores: dict[str, str]


@dataclass(frozen=True, slots=True)
class GTExEntitySettings:
    entities_path: str | None = None
    entities_file_type: Literal["csv", "txt"] = "csv"


@dataclass(frozen=True, slots=True)
class KnowledgeStoreConfig:
    retriever: dict[str, Any] | None = None
    document_store: dict[str, Any] | None = None
    embedder: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeGraphDBSettings:
    """Settings for the knowledge graph DB."""

    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeGraphGTExSettings:
    """Settings for the GTEx chat."""

    entities: GTExEntitySettings | None = None
    ner_path: str | None = None
    graph_db: KnowledgeGraphDBSettings | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeGraphSchemaSettings:
    """Settings for the knowledge graph's schema."""

    location: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeGraphSettings:
    """Settings for the knowledge graph."""

    schema: KnowledgeGraphSchemaSettings | None = None
    gtex: KnowledgeGraphGTExSettings | None = None


@dataclass(frozen=True, slots=True)
class Config(YAMLWizard):
    package: str
    paths: dict[str, str]
    vendors: list[str]
    knowledgestores: dict[str, KnowledgeStoreConfig]
    models: dict[str, Any]
    storages: dict[str, StorageConfig]
    agents: dict[str, AgentSettings]
    knowledge_graph: KnowledgeGraphSettings | None = None


@argsclass(frozen=True, slots=True)
class CLI:
    app_config: str = arg(default=os.getenv("APP_CONFIG"))
    app_workers: int = arg(default=os.getenv("APP_WORKERS", 1))
    app_host: str = arg(default=os.getenv("APP_HOST", "0.0.0.0"))
    app_port: int = arg(default=os.getenv("APP_PORT", 8000))
    app_subpath: str = arg(default=os.getenv("APP_SUBPATH", "/"), help="app subpath")
    storages: bool = arg(default=os.getenv("STORAGES", False))
    prompt_studio: bool = arg(
        default=os.getenv("PROMPTSTUDIO", False) in ("True", "true")
    )
    prompt_studio_subpath: str = arg(
        default=os.getenv("PROMPTSTUDIO_SUBPATH", "/playground")
    )
