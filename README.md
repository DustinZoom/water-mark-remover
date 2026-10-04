# Paraphraser

A small web app that rewrites text with the same meaning using a local LLM,
showing the original and the rewrite side by side. Output streams token by
token so CPU-only generation still feels responsive.

## Stack

- **Model server:** [Ollama](https://ollama.com), default model `qwen2.5:1.5b-instruct`
  (~1 GB, runs acceptably on CPU).
- **Backend:** FastAPI (`app/main.py`). `POST /api/rewrite` proxies to Ollama's
  streaming chat API and returns a chunked `text/plain` stream.
- **Frontend:** one static page (`app/static/index.html`), no build step. Reads
  the stream with `fetch` + `ReadableStream`.

## Run with Docker (recommended)

```bash
docker compose up -d --build
```

The first start downloads the model into a named volume, which takes a few
minutes. The app then serves on http://localhost:8000.

To use a different model, set `PARAPHRASE_MODEL` (any Ollama tag), e.g.
`PARAPHRASE_MODEL=llama3.2:3b docker compose up -d`.

## Run locally without Docker

Install Ollama, then:

```bash
ollama pull qwen2.5:1.5b-instruct
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Configuration

| Env var            | Default                    | Purpose                          |
|--------------------|----------------------------|----------------------------------|
| `OLLAMA_URL`       | `http://localhost:11434`   | Ollama endpoint                  |
| `PARAPHRASE_MODEL` | `qwen2.5:1.5b-instruct`    | Model tag                        |
| `MAX_INPUT_CHARS`  | `8000`                     | Input length limit               |

## API

- `GET /api/health`: whether Ollama is reachable and the model is pulled.
- `POST /api/rewrite` with `{"text": "..."}`: streams the rewrite as plain text.
  Returns 503 if Ollama is down and 422 for empty or oversized input.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests mock Ollama with [respx](https://lundberg.github.io/respx/), so they
need no model or Docker. They cover token streaming, upstream failures (503/502
and mid-stream errors), input validation, and the health check. CI
(`.github/workflows/ci.yml`) runs them on every push and PR, and also checks
that the Docker image builds.

## Not yet included

Quality metrics (semantic similarity), auth, and history.
