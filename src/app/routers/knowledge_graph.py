from typing import Annotated, Any
from fastapi import APIRouter, Body, Request
from fastapi.exceptions import HTTPException

from src.config.types import Memory, Message
from ..types import ChatResponse
import uuid

router = APIRouter()


@router.post("/knowledge-graph/chat", response_model=ChatResponse)
async def knowledge_graph_chat_simple(
    request: Request, message: Annotated[dict[str, Any], Body(embed=True)]
) -> ChatResponse:
    """Simplified knowledge graph chat endpoint without project/memory requirements."""

    # Get state from app
    state = request.app.state

    # Use default user if no auth header provided
    default_user = "system@invisibl.io"

    # Use consistent memory ID for session continuity
    session_memory_id = "session_default"

    # Enrich message
    message["origin"] = default_user
    message["memory_id"] = session_memory_id

    # Get configuration
    kg_settings = state.config.knowledge_graph
    if not kg_settings:
        raise HTTPException(
            status_code=500, detail="Knowledge graph configuration not found"
        )

    schema_settings = kg_settings.schema
    gtex_settings = kg_settings.gtex

    if not schema_settings or not schema_settings.location:
        raise HTTPException(status_code=500, detail="Schema location not configured")

    if not gtex_settings or not gtex_settings.entities or not gtex_settings.ner_path:
        raise HTTPException(status_code=500, detail="GTEx configuration not found")

    # Validate message content
    if not message.get("content"):
        raise HTTPException(status_code=422, detail="message.content cannot be empty")

    # Import here to avoid circular imports
    from src.plant import Plant

    # Create plant instance
    plant = Plant(state)

    # Get chat client and process
    chat_client = plant.get_chat_client()

    try:
        response = chat_client.gtex_chat(
            Message.from_dict(message),
            schema_path=schema_settings.location,
            entities_path=gtex_settings.entities.entities_path,
            entities_file_type=gtex_settings.entities.entities_file_type,
            model_path=gtex_settings.ner_path,
        )

        return ChatResponse(reply=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
