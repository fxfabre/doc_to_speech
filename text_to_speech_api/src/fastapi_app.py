import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi_utils.timing import add_timing_middleware
from starlette.responses import JSONResponse
from starlette.responses import RedirectResponse

from src.controllers import router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text To Speech API",
    version="1.0",
    description="From text to voice",
    default_response_class=JSONResponse,
    on_startup=[
        lambda: logger.info("FastAPI ready")
    ],
)
add_timing_middleware(app, record=logger.info, prefix="app")
app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=302)


@app.exception_handler(Exception)
async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Error during request %s", request.url, exc_info=exc)
    return JSONResponse(
        {
            "Type": type(exc).__name__,
            "Error": str(exc.args),
        },
        status_code=500,
    )
