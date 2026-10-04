import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def ndjson(*chunks):
    return "".join(json.dumps(c) + "\n" for c in chunks).encode()


def token(text, done=False):
    return {"message": {"role": "assistant", "content": text}, "done": done}


@pytest.fixture
def ollama():
    # Only Ollama's URL is mocked; TestClient talks to the app in-process.
    with respx.mock(base_url=main.OLLAMA_URL, assert_all_called=False) as mock:
        yield mock


def test_rewrite_streams_tokens(ollama):
    route = ollama.post("/api/chat").respond(
        200, content=ndjson(token("The cat "), token("sat down."), token("", done=True))
    )

    r = client.post("/api/rewrite", json={"text": "  A cat sat.  "})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == "The cat sat down."
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == main.MODEL
    assert sent["stream"] is True
    assert sent["messages"][0] == {"role": "system", "content": main.SYSTEM_PROMPT}
    assert sent["messages"][1] == {"role": "user", "content": "A cat sat."}


def test_rewrite_stops_at_done(ollama):
    ollama.post("/api/chat").respond(
        200, content=ndjson(token("Hi.", done=True), token(" trailing junk"))
    )
    assert client.post("/api/rewrite", json={"text": "Hello."}).text == "Hi."


def test_rewrite_surfaces_midstream_error(ollama):
    ollama.post("/api/chat").respond(
        200, content=ndjson(token("Partial"), {"error": "out of memory"})
    )
    r = client.post("/api/rewrite", json={"text": "Hello."})
    assert r.status_code == 200
    assert r.text == "Partial\n[error: out of memory]"


def test_rewrite_ollama_unreachable(ollama):
    ollama.post("/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    r = client.post("/api/rewrite", json={"text": "Hello."})
    assert r.status_code == 503
    assert "unreachable" in r.json()["detail"]


def test_rewrite_upstream_error(ollama):
    ollama.post("/api/chat").respond(404, json={"error": "model not found"})
    r = client.post("/api/rewrite", json={"text": "Hello."})
    assert r.status_code == 502
    assert "model not found" in r.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{}, {"text": ""}, {"text": "   \n "}, {"text": "a" * (main.MAX_CHARS + 1)}],
    ids=["missing", "empty", "whitespace", "too-long"],
)
def test_rewrite_rejects_bad_input(ollama, body):
    route = ollama.post("/api/chat")
    r = client.post("/api/rewrite", json=body)
    assert r.status_code == 422
    assert not route.called


def test_health_ok(ollama):
    ollama.get("/api/tags").respond(200, json={"models": [{"name": main.MODEL}]})
    assert client.get("/api/health").json() == {
        "ok": True, "model": main.MODEL, "max_chars": main.MAX_CHARS
    }


def test_health_model_missing(ollama):
    ollama.get("/api/tags").respond(200, json={"models": [{"name": "other:latest"}]})
    assert client.get("/api/health").json()["ok"] is False


def test_health_ollama_down(ollama):
    ollama.get("/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    body = client.get("/api/health").json()
    assert body["ok"] is False
    assert body["error"] == "Ollama unreachable"


def test_index_served_uncached():
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>Paraphraser</title>" in r.text
    assert r.headers["cache-control"] == "no-cache"
