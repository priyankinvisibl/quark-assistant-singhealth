from typing import Annotated, Any
from fastapi import APIRouter, Body
from fastapi.exceptions import HTTPException

from src.config.types import Message
from src.utils import read_yaml
from ..types import ChatResponse

router = APIRouter()


@router.post("/knowledge-graph/chat")
async def gtex_chat(message: Annotated[dict[str, Any], Body(embed=True)]):
    """Chat with GTEx data as context.

    This endpoint integrates GTEx data for generating responses.
    """
    message["origin"] = "default_user"
    message["memory_id"] = "default_memory"

    # Get the schema and entities paths from the configuration.
    raise_schema_exc, raise_gtex_exc = True, True
    schema_location, entities_path, entities_file_type, ner_path = "", "", "csv", ""

    # Use hardcoded config or load from file
    from src.config.types import Config

    try:
        # Load config using YAMLWizard with relative path
        config = Config.from_yaml_file("tmp/config.yaml")
        kg_settings = config.knowledge_graph
        if kg_settings is not None:
            schema_settings = kg_settings.schema
            gtex_settings = kg_settings.gtex
            if schema_settings is not None:
                schema_location = schema_settings.location
                raise_schema_exc = False
            if gtex_settings is not None:
                entities_settings = gtex_settings.entities
                if entities_settings is not None:
                    ner_path = gtex_settings.ner_path
                    entities_path = entities_settings.entities_path
                    entities_file_type = entities_settings.entities_file_type
                    if entities_path is not None:
                        raise_gtex_exc = False
    except Exception as e:
        # Debug: show what went wrong
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")

    if raise_schema_exc or not schema_location:
        raise HTTPException(
            status_code=500,
            detail=(
                "The schema location config has not been set; please contact your "
                "Quark administrator"
            ),
        )
    if raise_gtex_exc or not entities_path or not ner_path:
        raise HTTPException(
            status_code=500,
            detail=(
                "The GTEx config has not been set; please contact your Quark "
                "administrator"
            ),
        )

    # Create a simple chat client
    from src.pipelines.chat import Chat

    try:
        chat_client = Chat(config, None, None)  # No memory or knowledge store clients

        result = chat_client.gtex_chat(
            Message.from_dict(message),
            schema_path=schema_location,
            entities_path=entities_path,
            entities_file_type=entities_file_type,
            model_path=ner_path,
        )

        # Return only content, timestamp, and message_id
        return {
            "content": result.content,
            "timestamp": result.timestamp,
            "message_id": result.message_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
