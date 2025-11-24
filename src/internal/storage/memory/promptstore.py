from typing import Dict, Any, List
from src.internal.storage.types import PromptStoreClient
from src.config.types import PromptModel, ResponseModel, ResponseCollection
from .client import InMemoryClient
import uuid

class InMemoryQClient(PromptStoreClient):
    def __init__(self, client: InMemoryClient):
        self.client = client
        self.prompts_index = "prompts"
    
    def get_record(self, path: str, filename: str = None) -> Any:
        # For S3-like operations, return mock data
        return f"Mock content for {path}"
    
    def create_prompt(self, model: PromptModel) -> None:
        prompt_data = model.to_dict()
        prompt_id = str(uuid.uuid4())
        self.client.index_record(self.prompts_index, prompt_data, prompt_id)
    
    def get_prompt_collection(self, project: str, user: str, collection: str) -> ResponseModel:
        query = {"metadata.project": project, "metadata.owner": user, "metadata.collection": collection}
        result = self.client.search(self.prompts_index, query, size=100)
        
        prompts = []
        for hit in result["hits"]["hits"]:
            prompts.append(PromptModel.from_dict(hit["_source"]))
        
        return ResponseModel(
            title="Prompt Collection",
            data=prompts,
            metrics={"total_count": len(prompts)},
            collections={}
        )
    
    def get_prompt(self, project: str, user: str, name: str) -> PromptModel:
        query = {"metadata.project": project, "metadata.owner": user, "metadata.name": name}
        result = self.client.search(self.prompts_index, query, size=1)
        if result["hits"]["hits"]:
            return PromptModel.from_dict(result["hits"]["hits"][0]["_source"])
        return None
    
    def get_prompts(self, user: str, project: str, page: int = 0, size: int = 10) -> List[ResponseCollection]:
        query = {"metadata.owner": user, "metadata.project": project}
        from_ = page * size
        result = self.client.search(self.prompts_index, query, size, from_)
        
        prompts = []
        for hit in result["hits"]["hits"]:
            prompts.append(PromptModel.from_dict(hit["_source"]))
        
        return [ResponseCollection(
            title="Prompts",
            items=prompts,
            metrics={"total_count": result["hits"]["total"]["value"]},
            collections={}
        )]
    
    def update_prompt(self, id_: str, prompt: PromptModel) -> None:
        prompt_data = prompt.to_dict()
        self.client.update_record(self.prompts_index, id_, prompt_data)
    
    def delete_prompt(self, id_: str) -> None:
        self.client.delete_record(self.prompts_index, id_)
