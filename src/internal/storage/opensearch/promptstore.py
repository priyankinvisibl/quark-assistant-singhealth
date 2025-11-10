from typing import override

from src import jsunder
from src.config.types import PromptModel, ResponseCollection, ResponseModel
from src.internal.storage.opensearch.client import QClient as storage
from src.internal.storage.types import PromptStoreClient


# TODO: add exceptions
# thin wrapper class for promptstore over storage
class QClient(PromptStoreClient):
    def __init__(self, client: storage):
        self.client = client

    @override
    def create_prompt(self, prompt: PromptModel) -> None:
        self.client.upsert_record(
            "prompts", prompt.metadata.id, body=prompt.model_dump()
        )
        return None

    @override
    def get_prompt(self, project: str, user: str, uid: str) -> PromptModel | None:
        res = self.client.get_record("prompts", uid)
        if res:
            return PromptModel(**res.source)
        return None

    @override
    def get_prompt_collection(
        self,
        project: str,
        user: str,
        collection: str,
        size: int = 1,
        page: int = 1,
    ) -> ResponseModel:
        body = {
            "sort": [{"metadata.createdAt": {"order": "desc"}}],
            "size": size,
            "from": size * (int(page) - 1),
            "query": {
                "bool": {"filter": [{"term": {"metadata.collection": collection}}]}
            },
            "aggs": {
                "versions": {"terms": {"field": "metadata.version", "size": 10000}},
                "totalCount": {"cardinality": {"field": "metadata.collection"}},
            },
        }
        res = self.client.get_records("prompts", body=body)
        return ResponseCollection(
            title="Prompt Collection",
            items=[PromptModel(**response.source) for response in res.hits.hits],
            metrics={"totalCount": res.aggregations["totalCount"].metrics},
            collections={
                "filters": jsunder.get_values_buckets(
                    res.aggregations["versions"].buckets, "key"
                ),
            },
        )

    @override
    def get_prompts(
        self,
        user: str,
        project: str,
        page: int = 1,
        size: int = 10,
    ) -> list[ResponseCollection]:
        body = {
            "sort": [{"metadata.createdAt": {"order": "desc"}}],
            "size": size,
            "from": size * (int(page) - 1),
            "query": {"bool": {"filter": [{"term": {"metadata.owner": user}}]}},
            "collapse": {"field": "metadata.collection"},
            "aggs": {
                "totalCount": {"cardinality": {"field": "metadata.collection"}},
            },
        }
        res = self.client.get_records("prompts", body=body)
        return ResponseCollection(
            title="Prompt Collections",
            items=[PromptModel(**response.source) for response in res.hits.hits],
            metrics={"totalCount": res.aggregations["totalCount"].metrics},
            collections={},
        )

    @override
    def update_prompt(self, id_: str, prompt: PromptModel) -> None:
        self.client.upsert_record("prompts", id_, body=prompt.model_dump())
        return None

    @override
    def delete_prompt(self, id_: str) -> None:
        return self.client.delete_record("prompts", id_)
