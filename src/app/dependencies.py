import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request, exception_handlers
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from src.config.logging import logger
from src.config.types import User
from src.plant import Plant


# dependencies
# TODO: to expand
@dataclass(slots=True)
class Context:
    request: Request
    user: User
    project: str
    plant: Plant


class ContextDependency:
    def __init__(self):
        pass

    async def __call__(
        self,
        request: Request,
        gravity_userid: Annotated[str, Header()],
        project: str,
    ) -> Context:
        state = request.app.state
        # TODO: platform auth needs to be integrated
        try:
            user = next(user for user in state.users if user.email == gravity_userid)
        except KeyError:
            raise HTTPException(status_code=401)
        return Context(request, user, project, Plant(state))


# logging middleware
async def logger_middleware(request: Request, call_next: callable) -> JSONResponse:
    url = (
        f"{request.url.path}?{request.query_params}"
        if request.query_params
        else request.url.path
    )
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    port = getattr(client, "port", None)
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(e)
        response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
    finally:
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}"
        logger.info(
            f"{response.status_code} | {formatted_process_time}ms | {host}:{port} | {request.method} | {url}"
        )
        return response


# exception handlers
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    body = await request.body()
    query_params = request.query_params._dict
    detail = {
        "errors": exc.errors(),
        "body": body.decode(),
        "query_params": query_params,
    }
    logger.error(detail)
    return await exception_handlers.request_validation_exception_handler(request, exc)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    port = getattr(client, "port", None)
    detail = {
        "msg": exc.detail,
        "status_code": exc.status_code,
    }
    logger.error(detail)
    return await exception_handlers.http_exception_handler(request, exc)


exceptions = {
    RequestValidationError: request_validation_exception_handler,
    HTTPException: http_exception_handler,
}
middleware = logger_middleware
context = Annotated[Context, Depends(ContextDependency())]
dependencies = []
# type alias
# TODO
# async def auth_middleware()
