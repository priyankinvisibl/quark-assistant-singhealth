from typing import Any

from haystack import component, default_to_dict

from src.config.types import Message


# GENERALIZE TO ANY OBJECT
@component
class DictToMessageAdapter:
    @component.output_types(message=Message)
    def run(self, d: dict[str, Any]):
        return {"message": Message(**d)}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)
