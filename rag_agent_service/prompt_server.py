"""
RAG Prompt Editor API Server
============================
A lightweight FastAPI server for editing the live rag_agent_service system prompt.

Usage:
    cd rag_agent_service
    python prompt_server.py          # starts on http://0.0.0.0:8102
    python prompt_server.py --port 9003
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from prompt_store import (
    ensure_prompt_file,
    get_default_prompt_text,
    get_prompt_metadata,
    read_active_prompt,
    reset_active_prompt,
    write_active_prompt,
)


_THIS_DIR = Path(__file__).resolve().parent
_HTML_FILE = _THIS_DIR / "prompt_editor.html"

app = FastAPI(title="RAG Prompt Editor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SavePromptRequest(BaseModel):
    content: str


@app.get("/")
async def serve_ui():
    if not _HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="prompt_editor.html not found")
    return FileResponse(_HTML_FILE, media_type="text/html")


@app.get("/api/prompt")
async def get_prompt() -> JSONResponse:
    ensure_prompt_file()
    metadata = get_prompt_metadata()
    return JSONResponse(
        {
            "prompt": {
                "content": read_active_prompt(),
                **metadata,
            },
            "default_content": get_default_prompt_text(),
        }
    )


@app.put("/api/prompt")
async def save_prompt(body: SavePromptRequest) -> JSONResponse:
    try:
        path = write_active_prompt(body.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse(
        {
            "status": "ok",
            "chars": len(str(body.content or "")),
            "path": str(path),
        }
    )


@app.post("/api/prompt/reset")
async def reset_prompt() -> JSONResponse:
    content = reset_active_prompt()
    metadata = get_prompt_metadata()
    return JSONResponse(
        {
            "status": "ok",
            "prompt": {
                "content": content,
                **metadata,
            },
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG prompt editor server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8102, help="Bind port")
    args = parser.parse_args()
    print(f"  RAG Prompt Editor: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
