from typing import List, Tuple, Dict, Any
from src.internal.storage.types import MemoryClient, Memory, Message
from src.config.types import ResponseCollection
from .client import InMemoryClient
import uuid

class InMemoryMemClient(MemoryClient):
    def __init__(self, client: InMemoryClient):
        self.client = client
        self.memories_index = "memories"
        self.messages_index = "messages"
    
    def get_memories(self, user: str, size: int = 10, page: int = 1, 
                    sort: str = "desc", search: str = "", get_archived: bool = False,
                    filters: Dict = None) -> ResponseCollection:
        
        query = {"user": user}
        if filters:
            query.update(filters)
        
        from_ = (page - 1) * size
        sort_config = [{"field": "timestamp", "order": sort}]
        
        result = self.client.search(self.memories_index, query, size, from_, sort_config)
        
        memories = []
        for hit in result["hits"]["hits"]:
            memory_data = hit["_source"]
            memory_data["memory_id"] = hit["_id"]
            memories.append(Memory.from_dict(memory_data))
        
        return ResponseCollection(
            title="Memories",
            items=memories,
            metrics={"total_count": result["hits"]["total"]["value"]},
            collections={}
        )
    
    def get_memory(self, id_: str) -> Memory:
        result = self.client.search(self.memories_index, {"memory_id": id_}, size=1)
        if result["hits"]["hits"]:
            memory_data = result["hits"]["hits"][0]["_source"]
            memory_data["memory_id"] = id_
            return Memory.from_dict(memory_data)
        return None
    
    def create_memory(self, memory: Memory) -> str:
        memory_dict = memory.to_dict()
        memory_id = memory_dict.pop("memory_id", str(uuid.uuid4()))
        return self.client.index_record(self.memories_index, memory_dict, memory_id)
    
    def update_memory(self, memory_id: str, memory: Memory) -> Memory:
        memory_dict = memory.to_dict()
        memory_dict.pop("memory_id", None)
        self.client.update_record(self.memories_index, memory_id, memory_dict)
        return memory
    
    def delete_memory(self, memory_id: str):
        self.client.delete_record(self.memories_index, memory_id)
        # Also delete associated messages
        messages_result = self.client.search(self.messages_index, {"memory_id": memory_id}, size=1000)
        for hit in messages_result["hits"]["hits"]:
            self.client.delete_record(self.messages_index, hit["_id"])
    
    def get_messages(self, memory_id: str, size: int = 10, page: int = 1,
                    sort: str = "asc", search: str = "") -> ResponseCollection:
        
        from_ = (page - 1) * size
        sort_config = [{"field": "timestamp", "order": sort}]
        
        result = self.client.search(self.messages_index, {"memory_id": memory_id}, size, from_, sort_config)
        
        messages = []
        for hit in result["hits"]["hits"]:
            message_data = hit["_source"]
            message_data["message_id"] = hit["_id"]
            messages.append(Message.from_dict(message_data))
        
        return ResponseCollection(
            title="Messages",
            items=messages,
            metrics={"total_count": result["hits"]["total"]["value"]},
            collections={}
        )
    
    def add_messages(self, message: Message) -> str:
        message_dict = message.to_dict()
        message_id = message_dict.pop("message_id", str(uuid.uuid4()))
        return self.client.index_record(self.messages_index, message_dict, message_id)
    
    def get_relevant_history(self, memory_id: str, query: str):
        """Get relevant conversation history for a query."""
        messages, memory = self.get_history(memory_id)
        # Simple implementation - return last 5 messages
        return messages[-5:] if messages else []
    
    def get_history(self, memory_id: str) -> Tuple[List[Message], Memory]:
        # Get memory
        memory = self.get_memory(memory_id)
        
        # Get messages
        messages_result = self.client.search(
            self.messages_index, 
            {"memory_id": memory_id}, 
            size=1000,
            sort=[{"field": "timestamp", "order": "asc"}]
        )
        
        messages = []
        for hit in messages_result["hits"]["hits"]:
            message_data = hit["_source"]
            message_data["message_id"] = hit["_id"]
            messages.append(Message.from_dict(message_data))
        
        return messages, memory
    
    def download(self, memory_id: str) -> ResponseCollection:
        messages, memory = self.get_history(memory_id)
        data = {
            "memory": memory.to_dict() if memory else None,
            "messages": [m.to_dict() for m in messages]
        }
        return ResponseCollection(
            title="Download",
            items=[data],
            metrics={"total_count": 1},
            collections={}
        )
