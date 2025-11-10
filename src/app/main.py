from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from datargs import parse
from dotenv import load_dotenv
from fastapi import FastAPI, status

from src.app.dependencies import dependencies, exceptions, middleware
from src.app.routers import simple_chat
from src.config.state import State
from src.config.types import CLI

load_dotenv(override=True)

cli = parse(CLI)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state = State.from_config(cli)
    yield


app = FastAPI(
    lifespan=lifespan,
    docs_url=cli.app_subpath + "/docs",
    openapi_url=cli.app_subpath + "/openapi.json",
)


# download models and store if not already present
@app.get("/ping", status_code=status.HTTP_200_OK, include_in_schema=False)
async def ping():
    return "ok"


app.middleware("http")(middleware)


# TODO
# @router.post("/code/generate")
# @version(opts.app_version)
# async def generate_code(request: Request):
#     request_body = await request.json()
#     return executor.code(request_body["input"])


# Only include the simplified chat router
app.include_router(simple_chat.router, prefix=cli.app_subpath + "/api/v1")


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    uvicorn.run(
        app="src.app.main:app",
        # reload=True,
        workers=cli.app_workers,
        host=cli.app_host,
        port=cli.app_port,
        access_log=False,
    )
