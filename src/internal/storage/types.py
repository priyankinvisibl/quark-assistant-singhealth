from abc import ABC, abstractmethod
from typing import Any

from src.config.types import (
    KnowledgeBase,
    Memory,
    Message,
    PromptModel,
    ResponseCollection,
    ResponseModel,
)


class Storage(ABC):
    @abstractmethod
    def get_record(self, store: str, id_: str):
        pass

    @abstractmethod
    def upsert_record(self, store: str, id: str, content=dict[str, any]):
        pass

    @abstractmethod
    def store_exists(self, store: str):
        pass

    @abstractmethod
    def delete_record(self, store: str, id: str):
        pass

    @abstractmethod
    def get_records(self, store: str, id: str, content=dict[str, any]):
        pass


class MemoryClient(ABC):
    @abstractmethod
    def create_memory(self, memory: Memory) -> None:
        pass

    @abstractmethod
    def update_memory(self, id_: str, memory: Memory) -> None:
        pass

    @abstractmethod
    def delete_memory(self, id_: str) -> None:
        pass

    @abstractmethod
    def get_memories(
        self,
        user: str,
        size: int,
        page: int,
        sort: str = "",
        search: str = "",
        get_archived: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> ResponseCollection:
        pass

    @abstractmethod
    def add_messages(self, message: Message) -> None:
        pass

    @abstractmethod
    def get_memory(self, id_: str) -> Memory:
        pass

    @abstractmethod
    def get_relevant_history(self, memory_id: str, query: str):
        pass

    @abstractmethod
    def download(self, memory_id: str) -> ResponseCollection:
        pass

    # TODO: deprecrated
    @abstractmethod
    def get_history(self, memory_id: str) -> tuple[list[Message] | None, Memory | None]:
        pass

    @abstractmethod
    def get_messages(
        self,
        memory_id: str,
        size: int,
        page: int,
        sort: str = "",
        search: str = "",
    ) -> ResponseCollection:
        pass


class PromptStoreClient(ABC):
    @abstractmethod
    def create_prompt(self, model: PromptModel) -> None:
        pass

    @abstractmethod
    def get_prompt_collection(
        self, project: str, user: str, collection: str
    ) -> ResponseModel:
        pass

    @abstractmethod
    def get_prompt(self, project: str, user: str, name: str) -> PromptModel:
        pass

    @abstractmethod
    def get_prompts(
        self,
        user: str,
        project: str,
        page: int = 0,
        size: int = 10,
    ) -> list[ResponseCollection]:
        pass

    @abstractmethod
    def update_prompt(self, id_: str, prompt: PromptModel) -> None:
        pass

    @abstractmethod
    def delete_prompt(self, id_: str) -> None:
        pass


class KnowledgeStoreClient(ABC):
    @abstractmethod
    def create_knowledgebase(self, kb: KnowledgeBase) -> None:
        pass

    @abstractmethod
    def get_knowledgebase(self, id_: str) -> KnowledgeBase:
        pass

    @abstractmethod
    def get_knowledgebases(self, user: str, project: str) -> list[KnowledgeBase]:
        pass

    @abstractmethod
    def update_knowledgebase(self, id_: str, kb: KnowledgeBase) -> None:
        pass

    @abstractmethod
    def delete_knowledgebase(self, id_: str) -> None:
        pass

    @abstractmethod
    def get_document_store(self):
        pass

    @abstractmethod
    def get_retriever(self, retriever: str, store: str):
        pass

    def get_embedder(self, embedder: str):
        if embedder:
            from haystack.components.embedders import (
                SentenceTransformersDocumentEmbedder,
            )

            return SentenceTransformersDocumentEmbedder()
            # from haystack_integrations.components.embedders.optimum import OptimumDocumentEmbedder
            # from haystack.components.embedders import AzureOpenAIDocumentEmbedder
