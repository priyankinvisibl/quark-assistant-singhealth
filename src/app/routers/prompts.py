from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException

from src.app.dependencies import context
from src.config.types import PromptModel, PromptRecord, PromptVariable
from src.log_debug.analyzer import analyze_log_with_custom_bedrock
from src.log_debug.AnalyzeRequest import AnalyzeRequest

router = APIRouter()


# Prompts
@router.post(
    "/",
    status_code=201,
    tags=["prompts"],
)
async def create_prompt(
    ctx: context,
    name: Annotated[str, Body(embed=True)],
    collection: Annotated[str, Body(embed=True)],
    version: Annotated[str, Body(embed=True)],
    prompts: Annotated[list[PromptRecord], Body(embed=True)],
    description: Annotated[str, Body(embed=True)] | None = None,
    variables: Annotated[list[PromptVariable] | None, Body(embed=True)] = None,
):
    prompt = PromptModel.from_api(
        name,
        collection,
        description,
        ctx.project,
        ctx.user.name,
        version,
        prompts,
        variables,
    )
    ctx.plant.get_promptstore_client().create_prompt(prompt)
    return {"status": "created", "prompt": prompt}


@router.post(
    "/collections",
    tags=["prompts"],
)
async def get_prompts(
    ctx: context,
    parameters: Annotated[dict[str, Any], Body(embed=True)] = {},
):
    return ctx.plant.get_promptstore_client().get_prompts(
        ctx.user.name, ctx.project, **parameters
    )


@router.post(
    "/collections/{collection}",
    tags=["prompts"],
)
async def get_prompt_collection(
    ctx: context,
    collection: str,
    parameters: Annotated[dict[str, Any], Body(embed=True)] = {},
    id: str | None = None,
):
    return ctx.plant.get_promptstore_client().get_prompt_collection(
        ctx.project,
        ctx.user.name,
        collection,
        **parameters,
    )


@router.get(
    "/{id}",
    tags=["prompts"],
)
async def get_prompt(
    ctx: context,
    id: str,
):
    return ctx.plant.get_promptstore_client().get_prompt(
        ctx.project,
        ctx.user.name,
        id,
    )


@router.post(
    "/{id}/run",
    tags=["prompts"],
)
async def run_prompt(
    ctx: context,
    id: str,
    provider: Annotated[str, Body(embed=True)],
    config: Annotated[dict[str, Any], Body(embed=True)],
    values: Annotated[dict[str, Any], Body(embed=True)],
    message: Annotated[dict[str, Any], Body(embed=True)],
):
    prompt = ctx.plant.get_promptstore_client().get_prompt(
        ctx.project,
        ctx.user.name,
        id,
    )
    if prompt:
        builder = prompt.get_builder()
    generator = ctx.plant.get_chat_generator(provider, config)
    return ctx.plant.run_components(
        {"builder": builder, "generator": generator}, values, message
    )


# TODO
# @router.put(
#     "/{id}",
#     status_code=204,
#     tags=["prompts"],
# )
# async def update_prompt(
#     ctx: context,
#     id: str,
#     prompt: PromptModel,
# ):

#     ctx.plant.get_promptstore_client().update_prompt(id, prompt)
#     return {"status": "updated", "prompt": prompt}


@router.delete(
    "/{id}",
    tags=["prompts"],
)
async def delete_prompt(ctx: context, id: str):
    if ctx.plant.get_promptstore_client().delete_prompt(id) is None:
        return {"status": "deleted"}
    return {"status": "error deleting prompt"}

@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        result = analyze_log_with_custom_bedrock(
            request.error_log,
            request.pipeline_type
        )
        return {"analysis": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")