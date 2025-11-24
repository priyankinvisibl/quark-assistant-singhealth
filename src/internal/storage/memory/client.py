from typing import Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class InMemoryRecord:
    id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class InMemoryClient:
    def __init__(self, config=None):
        self.stores: Dict[str, List[InMemoryRecord]] = {}
        self.config = config
    
    def get_records(self, index: str, query: Dict = None) -> Any:
        if index not in self.stores:
            self.stores[index] = []
        
        records = self.stores[index]
        
        # Simple filtering
        if query:
            filtered = []
            for record in records:
                match = True
                for key, value in query.items():
                    if key not in record.data or record.data[key] != value:
                        match = False
                        break
                if match:
                    filtered.append(record)
            records = filtered
        
        # Mock OpenSearch response structure
        class MockHit:
            def __init__(self, record):
                self.source = {"object": record.data}
        
        class MockHits:
            def __init__(self, records):
                self.hits = [MockHit(r) for r in records]
                self.total = {"value": len(records)}
        
        class MockResponse:
            def __init__(self, records):
                self.hits = MockHits(records)
        
        return MockResponse(records)
    
    def index_record(self, index: str, data: Dict[str, Any], id: str = None) -> str:
        if index not in self.stores:
            self.stores[index] = []
        
        record_id = id or str(uuid.uuid4())
        record = InMemoryRecord(id=record_id, data=data)
        self.stores[index].append(record)
        return record_id
    
    def update_record(self, index: str, id: str, data: Dict[str, Any]):
        if index not in self.stores:
            return
        
        for record in self.stores[index]:
            if record.id == id:
                record.data.update(data)
                record.timestamp = datetime.now()
                break
    
    def delete_record(self, index: str, id: str):
        if index not in self.stores:
            return
        
        self.stores[index] = [r for r in self.stores[index] if r.id != id]
    
    def search(self, index: str, query: Dict = None, size: int = 10, from_: int = 0, sort: List = None) -> Dict:
        if index not in self.stores:
            self.stores[index] = []
        
        records = self.stores[index]
        
        # Apply filtering
        if query:
            filtered = []
            for record in records:
                match = True
                for key, value in query.items():
                    if key not in record.data or record.data[key] != value:
                        match = False
                        break
                if match:
                    filtered.append(record)
            records = filtered
        
        # Apply sorting
        if sort:
            for sort_field in reversed(sort):
                field_name = sort_field.get("field", "timestamp")
                reverse = sort_field.get("order", "asc") == "desc"
                records.sort(key=lambda x: getattr(x, field_name, x.data.get(field_name, "")), reverse=reverse)
        
        # Apply pagination
        total = len(records)
        records = records[from_:from_ + size]
        
        return {
            "hits": {
                "hits": [{"_source": r.data, "_id": r.id} for r in records],
                "total": {"value": total}
            }
        }
