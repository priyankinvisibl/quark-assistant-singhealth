import warnings
from typing import Annotated, Any

from fastapi import APIRouter, Body, Form, Response, UploadFile
from fastapi.exceptions import HTTPException

from src.app.dependencies import context
from src.config.types import Memory, Message, ResponseCollection

from ..types import ChatResponse

router = APIRouter()


# Memories
@router.get("/")
async def initiate_chat(ctx: context):
    return Memory.new(ctx.user.name)


@router.put("/{memory_id}", status_code=201)
async def update_memory(
    ctx: context, memory_id: str, memory: Annotated[dict[str, Any], Body()]
):
    return ctx.plant.get_memory_client().update_memory(
        memory_id, Memory.from_dict(memory)
    )


@router.post(
    "/", response_model=ResponseCollection, response_model_exclude_defaults=True
)
async def get_memories(
    ctx: context, parameters: Annotated[dict[str, Any], Body(embed=True)] = {}
):
    return ctx.plant.get_memory_client().get_memories(
        ctx.user.name,
        int(parameters.get("size", 10)),
        int(parameters.get("page", 1)),
        sort=parameters.get("sort", "desc"),
        search=parameters.get("search", ""),
        get_archived=parameters.get("getArchived", False),
        filters=parameters.get("filters"),
    )


@router.get("/{memory_id}")
async def get_memory(ctx: context, memory_id: str):
    return ctx.plant.get_memory_client().get_memory(memory_id)


@router.post(
    "/{memory_id}/messages",
    response_model=ResponseCollection,
    response_model_exclude_defaults=True,
)
async def get_messages(
    ctx: context,
    memory_id: str,
    parameters: Annotated[dict[str, Any], Body(embed=True)] = {},
):
    return ctx.plant.get_memory_client().get_messages(
        memory_id,
        int(parameters.get("size", 10)),
        int(parameters.get("page", 1)),
        sort=parameters.get("sort", "asc"),
        search=parameters.get("search", ""),
    )


@router.post("/{memory_id}/download")
async def download(
    ctx: context,
    memory_id: str,
):
    return ctx.plant.get_memory_client().download(memory_id)


@router.delete("/{memory_id}/")
async def delete(
    ctx: context,
    memory_id: str,
):
    return ctx.plant.get_memory_client().delete_memory(memory_id)


@router.post("/{memory_id}/chat")
async def chat(
    ctx: context,
    memory_id: str,
    message: Annotated[dict[str, Any], Body(embed=True)] = None,
):
    message["origin"] = ctx.user.name
    message["memory_id"] = memory_id
    return {"reply": ctx.plant.get_chat_client().chat(Message.from_dict(message))}


@router.post("/{memory_id}/summarize")
async def summarize(
    ctx: context,
    memory_id: str,
):
    return {"summary": ctx.plant.get_chat_client().summarize(memory_id, force=True)}


# TODO: integrate multimodal with regular chat
@router.post("/{memory_id}/chat/multimodal")
async def multimodal_chat(
    ctx: context,
    memory_id: str,
    file: UploadFile,
    content: str = Form(),
    # attachments: list[str] | None = None,
):
    """ """
    f = (file.content_type, await file.read(), file.filename)
    message = {"content": content, "origin": ctx.user.name, "memory_id": memory_id}
    message = {"content": content, "origin": ctx.user.name, "memory_id": memory_id}
    return {
        "reply": ctx.plant.get_chat_client().multimodal_chat(
            Message.from_dict(message),
            f,
        )
    }
    # raise HTTPException(status_code=400, detail="please upload a file or attach a link")


@router.post("/{memory_id}/image/multimodal")
async def multimodal_chat(
    ctx: context,
    memory_id: str,
    content: str = Form(),
    file: UploadFile = None,
    attachments: list[str] = None,
):
    """ """
    if file:
        f = (file.content_type, await file.read())
    if f[0] or attachments:
        message = {"content": content, "origin": ctx.user.name, "memory_id": memory_id}
        image_bytes = ctx.plant.get_chat_client().multimodal_image(
            Message.from_dict(message), f, attachments
        )
        return Response(content=image_bytes, media_type="image/png")
    raise HTTPException(status_code=400, detail="please upload a file or attach a link")


@router.post("/agents/generate")
async def run_agent(
    ctx: context,
    message: Annotated[dict[str, Any], Body(embed=True)],
):
    # Check if a memory exists for the given analysis ID.
    metadata = message.get("metadata", {})
    job_id = metadata.get("analysis", metadata.get("job", {})).get("id")
    if job_id is None or not job_id:
        raise HTTPException(
            status_code=422,
            detail="message.metadata.job/analysis.id is a required field",
        )
    agent = metadata.get("agent")
    if not agent:
        raise HTTPException(
            status_code=422, detail="message.metadata.agent is a required field"
        )
    analysis_memories = ctx.plant.get_memory_client().get_memories(
        ctx.user.name,
        10,
        1,
        sort="desc",
        search="",
        get_archived=False,
        filters={
            "metadata.analysis.id": job_id,
            "metadata.agent": agent,
        },
    )

    # If such a memory does exist, skip the agent call and return the latest message.
    if (
        analysis_memories.metrics.get("total_count", 0) > 0
        and len(analysis_memories.items) > 0
    ):
        memory_id = analysis_memories.items[0].memory_id
        if memory_id is None:
            warnings.warn("Existing memory object malformed; skipping use")
        else:
            memory = ctx.plant.get_memory_client().get_memory(id_=memory_id)
            messages = ctx.plant.get_memory_client().get_messages(
                memory_id=memory_id, size=2, page=1, sort="asc", search=""
            )
            if messages.metrics.get("total_count", 0) > 1 and len(messages.items) > 1:
                return {"response": messages.items[1]}
            if messages.metrics.get("total_count", 0) > 0 and len(messages.items) > 0:
                return {"response": messages.items[0]}

    # Otherwise, create a new memory and call the agent.
    memory = Memory.new(ctx.user.name)
    message["origin"] = memory.user
    message["memory_id"] = memory.memory_id

    message_obj = Message.from_dict(message)

    # message_obj.metadata["hidden"] = True
    # memory.metadata["archive"] = True
    memory.name = message_obj.metadata.get("analysis", {}).get("name", "")
    memory.metadata = message_obj.metadata
    if not message_obj.metadata.get("agent") or (
        not message_obj.content and not message_obj.metadata.get("document")
    ):
        raise HTTPException(
            status_code=422,
            detail="message.metadata.document OR message.content is a required field",
        )
    return {"response": ctx.plant.run_agent(message_obj, memory)}


@router.post("/{memory_id}/enrichment-analysis/chat", response_model=ChatResponse)
async def knowledge_graph_chat(
    ctx: context, memory_id: str, message: Annotated[dict[str, Any], Body(embed=True)]
) -> ChatResponse:
    """Chat with the knowledge graph as context.

    Currently used in the Quark Assistant in Insights after "Continue Conversation."
    """
    message["origin"] = ctx.user.name
    message["memory_id"] = memory_id

    # Get the graph and schema locations.
    raise_exc = True
    schema_location = ""
    kg_settings = ctx.plant.state.config.knowledge_graph
    if kg_settings is not None:
        schema_settings = kg_settings.schema
        if schema_settings is not None:
            schema_location = schema_settings.location
            raise_exc = False

    if raise_exc or not schema_location:
        raise HTTPException(
            status_code=500,
            detail=(
                "The schema location config has not been set; please contact your "
                "Quark administrator"
            ),
        )

    if message.get("content") is None:
        raise HTTPException(status_code=422, detail="message.content cannot be empty")

    graph_location = message.get("metadata", {}).get("document")
    if graph_location is None:
        raise HTTPException(
            status_code=422, detail="message.metadata.document is a required field"
        )

    return ChatResponse(
        reply=ctx.plant.get_chat_client().knowledge_graph_chat(
            Message.from_dict(message),
            graph_location=graph_location,
            schema_location=schema_location,
        )
    )


@router.post("/knowledge-graph/chat", response_model=ChatResponse)
async def gtex_chat(
    message: Annotated[dict[str, Any], Body(embed=True)]
) -> ChatResponse:
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
    from src.utils import read_yaml
    
    try:
        config_data = read_yaml("config/config.yaml")
        config = Config.from_dict(config_data)
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
    except Exception:
        pass  # Use defaults or raise exceptions below

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
    from src.config.types import Config
    
    config_data = read_yaml("config/config.yaml")
    config = Config.from_dict(config_data)
    
    chat_client = Chat(config, None, None)  # No memory or knowledge store clients
    
    return ChatResponse(
        reply=chat_client.gtex_chat(
            Message.from_dict(message),
            schema_path=schema_location,
            entities_path=entities_path,
            entities_file_type=entities_file_type,
            model_path=ner_path,
        )
    )
