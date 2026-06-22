"""
api.py — deployment entrypoint for the GraphRAG pulmonology assistant HTTP API.

Routes:
    GET  /         → service info
    GET  /health   → liveness/readiness probe (used by render.yaml)
    POST /chat     → { message, session_id?, user_id? } → { answer, session_id }

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000
On Render the start command + $PORT come from render.yaml (`uvicorn api:app`).

Design notes
------------
- ONE pipeline is built at startup and reused (constructing it opens Pinecone/
  Neo4j/Gemini clients — expensive to do per request).
- `GraphRAGPipeline.run()` is SYNCHRONOUS and its memory layer calls
  `asyncio.run()`, which cannot run inside a live event loop. So `/chat` is a
  plain `def` endpoint — FastAPI runs sync endpoints in a threadpool, off the
  event loop, which is exactly what the memory layer needs.
- The shared pipeline (its episodic event loop + client connections) is not
  proven thread-safe, so calls are serialized with a lock. For real concurrency
  run multiple uvicorn worker PROCESSES (each gets its own pipeline).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# ── Bootstrap: project root on path + Windows/Aura TLS + UTF-8 ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Route TLS through the OS trust store BEFORE any Pinecone/Neo4j/Gemini import.
# Required for Neo4j Aura (neo4j+s://) and Pinecone behind a proxy/AV/VPN
# ("unable to get local issuer certificate"). Falls back to certifi.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from graphrag.config.settings import ConfigError, settings

logger = logging.getLogger("api")


# ── Request / response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message/question.")
    session_id: str = Field("default", description="Conversation/memory key.")
    user_id: str | None = Field(
        None, description="Enables episodic (long-term) memory when provided."
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class SessionEndRequest(BaseModel):
    session_id: str = Field("default", description="The conversation/session to close.")
    user_id: str = Field(
        "", description="User id — episodic memory is per-user; empty = nothing stored."
    )


# ── App lifecycle: build the pipeline once ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings.validate_required("api")
    except ConfigError as e:
        raise RuntimeError(str(e)) from e

    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline

    app.state.pipeline = GraphRAGPipeline()
    app.state.lock = threading.Lock()
    try:
        yield
    finally:
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None:
            pipeline.close()


app = FastAPI(
    title="Enervera Pulmonology Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — from settings.CORS_ORIGINS (comma-separated, or "*").
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce X-API-Key only when API_KEY is configured; open otherwise."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Enervera Pulmonology Assistant", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream the answer as NDJSON: one validated UI block per line.

    Media type `application/x-ndjson`. The frontend reads the body stream, splits
    on "\\n", JSON.parses each complete line, and renders by `.type` as it arrives.
    """
    pipeline = request.app.state.pipeline
    lock = request.app.state.lock

    def ndjson_stream():
        # Sync generator → Starlette iterates it in a threadpool (off the event
        # loop), so the memory layer's asyncio.run() works. The lock is held for
        # the STREAM's whole duration: the shared pipeline isn't thread-safe
        # (scale with multiple worker processes instead).
        with lock:
            try:
                for block in pipeline.run(
                    query_text=req.message,
                    session_id=req.session_id,
                    user_id=req.user_id,
                ):
                    yield json.dumps(block, ensure_ascii=False) + "\n"
            except Exception as exc:  # never leak a stack trace; emit a block
                logger.exception("Pipeline error during /chat stream")
                yield json.dumps(
                    {
                        "type": "warning",
                        "data": {
                            "text": "Sorry — something went wrong generating a response. "
                            "Please try again.",
                            "severity": "info",
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")


@app.post("/session/end", dependencies=[Depends(require_api_key)])
def end_session(req: SessionEndRequest, request: Request) -> dict:
    """
    Close a conversation: consolidate the session and write ONE episode to
    episodic memory. Call this from the frontend when the chat closes.
    Episodic memory is updated ONLY here — never per message.
    """
    pipeline = request.app.state.pipeline
    lock = request.app.state.lock
    with lock:
        try:
            return pipeline.end_session(user_id=req.user_id, session_id=req.session_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Session-end error: {exc}") from exc


if __name__ == "__main__":
    # Convenience: `python api.py` starts the server (same as `uvicorn api:app`).
    # On Render the start command in render.yaml runs `uvicorn api:app` instead.
    import uvicorn

    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)

