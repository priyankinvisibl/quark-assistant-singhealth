from typing import override

from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

from src.config.types import KnowledgeBase
from src.internal.storage.opensearch.client import QClient
from src.internal.storage.types import KnowledgeStoreClient


# TODO: add exceptions
# thin wrapper class for knowledgestore over storage
class KSClient(KnowledgeStoreClient):
    def __init__(self, client: QClient):
        self.client = client
        self.document_store = OpenSearchDocumentStore()
        self.document_store._client = client.client

    @override
    def create_knowledgebase(self, kb: KnowledgeBase) -> None:
        self.client.upsert_record("knowledgebases", kb.id, body=kb.to_dict())
        return None

    @override
    def get_knowledgebase(self, id_: str) -> KnowledgeBase:
        response = self.client.get_record("knowledgebases", id_)
        if not response:
            return None
        return KnowledgeBase.from_dict(response.source)

    @override
    def get_knowledgebases(self, user: str, project: str) -> KnowledgeBase:
        res = self.client.get_records("knowledgebases").hits.hits
        return KnowledgeBase.from_list([d.source for d in res])

    @override
    def update_knowledgebase(self, id_: str, kb: KnowledgeBase) -> None:
        self.client.upsert_record("knowledgebases", id_, body=kb.to_dict())
        return None

    @override
    def delete_knowledgebase(self, id_: str) -> None:
        res = self.client.delete_record("knowledgebases", id_)
        return None

    # TODO: edge cases
    # TODO: refer to discussion in discord
    @override
    def get_document_store(self, store: list[str]):
        self.document_store._index = store
        return self.document_store

    @override
    def get_retriever(self, retriever: str, store: list[str]):
        self.document_store._index = ",".join(store)
        if retriever == "embedding":
            return OpenSearchEmbeddingRetriever(
                document_store=self.document_store, top_k=2
            )
        return OpenSearchBM25Retriever(document_store=self.document_store, top_k=2)

    @override
    def get_embedder(self, embedder: str):
        return super().get_embedder(embedder)
