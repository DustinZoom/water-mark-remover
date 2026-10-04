"""Semantic paraphraser: FastAPI front for a local Ollama model, streaming tokens."""

import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("PARAPHRASE_MODEL", "qwen2.5:1.5b-instruct")
MAX_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))
STATIC_DIR = Path(__file__).parent / "static"

SYSTEM_PROMPT = (
    "You are a paraphrasing tool. Rewrite the user's text so that it has exactly "
    "the same meaning, facts, and tone, but uses different wording and sentence "
    "structure. Keep paragraph breaks, names, numbers, and technical terms intact. "
    "Do not add, remove, or comment on information. Output only the rewritten text."
)

app = FastAPI(title="Paraphraser")


class RewriteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


@app.get("/")
async def index():
    # no-cache = revalidate each load, so redeploys show up without a hard refresh
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError:
        return {"ok": False, "model": MODEL, "error": "Ollama unreachable"}
    return {"ok": MODEL in models, "model": MODEL, "max_chars": MAX_CHARS}


@app.post("/api/rewrite")
async def rewrite(req: RewriteRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Text is empty.")

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192},
    }

    # Open the upstream stream before responding so connection/model errors
    # surface as a proper HTTP error instead of a half-written 200.
    client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None))
    try:
        upstream = await client.send(
            client.build_request("POST", f"{OLLAMA_URL}/api/chat", json=payload),
            stream=True,
        )
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(503, "Model server unreachable. Is Ollama running?")

    if upstream.status_code != 200:
        detail = (await upstream.aread()).decode(errors="replace")
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(502, f"Model server error: {detail}")

    async def tokens():
        try:
            async for line in upstream.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    yield f"\n[error: {chunk['error']}]"
                    break
                yield chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    break
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        tokens(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
