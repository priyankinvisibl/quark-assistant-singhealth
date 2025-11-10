from typing import Annotated, Any

from fastapi import APIRouter, Body

from src.app.dependencies import context
from src.config.types import KnowledgeBase

router = APIRouter()


# Knowledgebases
@router.post("/", status_code=201)
async def create_knowledgebase(
    ctx: context,
    knowledgebase: Annotated[dict[str, Any], Body(embed=True)],
):
    kb = KnowledgeBase.new(ctx.user.name, ctx.project, knowledgebase["name"])
    ctx.plant.get_knowledgestore_client().create_knowledgebase(kb)
    return {"status": "ok", "knowledgebase": kb}


@router.get("/")
async def get_knowledgebases(
    ctx: context,
):
    return ctx.plant.get_knowledgestore_client().get_knowledgebases(
        ctx.user.name, ctx.project
    )


@router.get("/{kb_id}")
async def get_knowledgebase(
    ctx: context,
    kb_id: str,
):
    return ctx.plant.get_knowledgestore_client().get_knowledgebase(kb_id)


# TODO: websocket?
@router.post("/{kb_id}/embed")
async def embed_document(
    ctx: context,
    kb_id: str,
    filename: Annotated[str, Body(embed=True)],
    embedding_type: Annotated[str, Body(embed=True)],
):
    return ctx.plant.get_embed_client().embed_file(
        filename,
        ctx.user.name,
        ctx.project,
        kb_id,
        embedding_type,
    )
