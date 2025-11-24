from typing import List, Dict, Any
from src.internal.storage.types import KnowledgeStoreClient
from src.config.types import ResponseCollection, KnowledgeBase
from .client import InMemoryClient
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
import uuid

class InMemoryKSClient(KnowledgeStoreClient):
    def __init__(self, client: InMemoryClient):
        self.client = client
        self.knowledgebases_index = "knowledgebases"
        self.embeddings_index = "embeddings"
        self.document_store = InMemoryDocumentStore()
    
    def get_knowledgebase(self, id_: str) -> KnowledgeBase:
        result = self.client.search(self.knowledgebases_index, {"id": id_}, size=1)
        if result["hits"]["hits"]:
            kb_data = result["hits"]["hits"][0]["_source"]
            return KnowledgeBase.from_dict(kb_data)
        return None
    
    def get_knowledgebases(self, user: str, project: str) -> List[KnowledgeBase]:
        query = {"owner": user, "project": project}
        result = self.client.search(self.knowledgebases_index, query, size=100)
        
        kbs = []
        for hit in result["hits"]["hits"]:
            kb_data = hit["_source"]
            kbs.append(KnowledgeBase.from_dict(kb_data))
        
        return kbs
    
    def get_retriever(self, retriever_type: str, store: str = None):
        if retriever_type == "bm25":
            return InMemoryBM25Retriever(document_store=self.document_store)
        return None
    
    def get_document_store(self):
        return self.document_store
    
    def create_knowledgebase(self, kb: KnowledgeBase) -> None:
        kb_data = kb.to_dict()
        kb_id = kb_data.get("id", str(uuid.uuid4()))
        self.client.index_record(self.knowledgebases_index, kb_data, kb_id)
    
    def update_knowledgebase(self, id_: str, kb: KnowledgeBase) -> None:
        kb_data = kb.to_dict()
        self.client.update_record(self.knowledgebases_index, id_, kb_data)
    
    def delete_knowledgebase(self, id_: str) -> None:
        self.client.delete_record(self.knowledgebases_index, id_)
    
    def get_embeddings(self, embedding_ids: List[str]) -> List[Dict]:
        embeddings = []
        for emb_id in embedding_ids:
            result = self.client.search(self.embeddings_index, {"embedding_id": emb_id}, size=1)
            if result["hits"]["hits"]:
                embeddings.append(result["hits"]["hits"][0]["_source"])
        return embeddings
