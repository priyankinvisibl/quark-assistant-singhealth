from pydantic import BaseModel

from src.config.types import Message


class ChatResponse(BaseModel):
    """The response model for a memory chat request."""

    reply: Message
