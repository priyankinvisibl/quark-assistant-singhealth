from typing import Any, override

from src.config.types import Memory, Message, ResponseCollection
from src.internal.storage.opensearch.client import QClient
from src.internal.storage.types import MemoryClient
from src.utils import get_zulu_time


# TODO: add exceptions, add user scope for better AC
# thin wrapper class for memory over storage
# move memories and messages keyword to config
class MemClient(MemoryClient):
    def __init__(self, client: QClient):
        self.client = client

    @override
    def create_memory(self, memory: Memory) -> None:
        self.client.upsert_record("memories", memory.memory_id, body=memory.to_dict())
        return None

    @override
    def get_memory(self, id_: str) -> Memory:
        response = self.client.get_record("memories", id_)
        if not response:
            return None
        return Memory.from_dict(response.source)

    @override
    def update_memory(self, id_: str, memory: Memory):
        memory.updated_at = get_zulu_time()
        self.client.upsert_record("memories", id_, body=memory.to_dict())

    @override
    def delete_memory(self, id_: str) -> None:
        self.client.delete_by_query(
            "messages",
            query={"query": {"term": {"memory_id": id_}}, "size": 1000},
        )
        self.client.delete_record("memories", id_)
        return None

    @override
    def get_memories(
        self,
        user: str,
        size: int = 10000,
        page: int = 1,
        sort: str = "desc",
        search: str = "",
        get_archived: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> ResponseCollection:
        page = size * (int(page) - 1)
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"user": user}},
                        {"term": {"metadata.archive": get_archived}},
                    ],
                }
            },
            "sort": [{"updated_at": {"order": sort}}],
            "size": size,
            "from": page,
            "aggs": {"total_count": {"cardinality": {"field": "memory_id"}}},
        }
        if filters is not None:
            for filter_key, filter_value in filters.items():
                query["query"]["bool"]["filter"].append(
                    {"term": {filter_key: filter_value}}
                )

        if search != "":
            query["query"]["bool"]["filter"].append({"match": {"name": search}})

        res = self.client.get_records(
            store="memories",
            body=query,
        )
        return ResponseCollection(
            title="Memories",
            items=Memory.from_list([d.source for d in res.hits.hits]),
            metrics={"total_count": res.aggregations["total_count"].metrics},
        )

    @override
    def add_messages(self, message: Message) -> None:
        self.client.upsert_record(
            "messages", message.message_id, body=message.to_dict()
        )
        return None

    @override
    def download(self, memory_id: str) -> ResponseCollection:
        res = self.client.scroll(
            "messages",
            query={
                "query": {"term": {"memory_id": memory_id}},
                "size": 1000,
                "sort": [{"timestamp": {"order": "desc"}}],
            },
        )
        if res:
            return ResponseCollection(
                title="Messages",
                items=Message.from_list([d.source for d in res.hits.hits]),
            )
        return None

    @override
    def get_messages(
        self,
        memory_id: str,
        size: int = 10000,
        page: int = 1,
        sort: str = "desc",
        search: str = "",
        get_hidden: bool = False,
    ) -> ResponseCollection:
        page = size * (int(page) - 1)
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"memory_id": memory_id}},
                    ]
                }
            },
            "sort": [{"timestamp": {"order": sort}}],
            "size": size,
            "from": page,
            "aggs": {"total_count": {"cardinality": {"field": "message_id"}}},
        }
        if search != "":
            query["query"]["bool"]["filter"].append({"match": {"content": search}})
        # TODO:
        if get_hidden:
            query["query"]["bool"]["filter"].append(
                {"term": {"metadata.hidden": get_hidden}}
            )
        res = self.client.get_records(store="messages", body=query)
        return ResponseCollection(
            title="Messages",
            items=Message.from_list([d.source for d in res.hits.hits]),
            metrics={"total_count": res.aggregations["total_count"].metrics},
        )

    # TODO: find correct use case
    @override
    def get_relevant_history(self, memory_id: str, query: str):
        memory = self.get_memory(memory_id)
        if not memory:
            return (None, None)

        history = self.get_messages(memory_id, size=10, query=query).items
        if len(history) <= 0:
            return (None, memory)
        return (history, memory)

    @override
    def get_history(self, memory_id: str) -> tuple[list[Message] | None, Memory | None]:
        memory = self.get_memory(memory_id)
        if not memory:
            return (None, None)
        history = self.get_messages(memory_id, size=10).items[::-1]
        summary = memory.metadata.get("summary", None)
        if summary:
            history.insert(
                0, Message(summary["content"], memory_id=memory_id, origin="assistant")
            )
        if len(history) <= 0:
            return (None, memory)
        return (history, memory)
