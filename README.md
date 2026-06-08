# qwen2api-rs

API-only gateway that exposes Qwen Web as OpenAI, Anthropic, Gemini, image, video, file, and embedding compatible endpoints.

This project is a Rust rewrite of the core gateway ideas from `YuJunZhiXue/qwen2API`. It intentionally does not ship a browser Web UI. The service is designed to be started on a server with Docker Compose and configured through a small `.env` file.

## Features

- OpenAI Chat Completions: `/v1/chat/completions`
- OpenAI Responses: `/v1/responses`
- Anthropic Messages: `/v1/messages`, `/anthropic/v1/messages`
- Gemini `generateContent` and `streamGenerateContent`
- OpenAI Images and Videos backed by Qwen `t2i` / `t2v`
- File upload and attachment forwarding
- Text-based tool calling for clients that expect function/tool calls
- Account pool with retries, rate-limit cooldowns, token refresh, and chat_id prewarming
- Health probes: `/healthz`, `/readyz`

## Configuration

Copy the example file and fill in the values you need:

```bash
cp .env.example .env
```

Minimal `.env`:

```env
API_KEY=sk-qwen2api-change-me
PORT=7860
DATA_DIR=./data
```

Provide Qwen accounts in one of two ways:

```json
[
  {
    "email": "user@example.com",
    "token": "token-from-chat.qwen.ai-localStorage",
    "password": "optional-used-for-token-refresh"
  }
]
```

Save that JSON as `data/accounts.json`, or pass the same JSON array through `QWEN_ACCOUNTS_JSON`.

Optional env values:

| Variable | Purpose | Default |
| --- | --- | --- |
| `API_KEY` | API key required by compatibility endpoints. | `change-me-now` |
| `PORT` | HTTP listen port. Docker uses host networking. | `7860` |
| `DATA_DIR` | Persistent state directory. | `./data` |
| `QWEN_ACCOUNTS_JSON` | Inline JSON array of Qwen accounts. | unset |
| `ACCOUNTS_FILE` | Custom account JSON path. | `$DATA_DIR/accounts.json` |
| `UPSTREAM_PROXY` | Outbound proxy for `chat.qwen.ai`. `HTTP_PROXY` / `HTTPS_PROXY` also work. | unset |
| `DEFAULT_MODEL` | Fallback Qwen model for unknown aliases. | `qwen3.7-plus` |
| `LOG_LEVEL` | `trace`, `debug`, `info`, `warn`, or `error`. | `info` |

All pool, retry, media, context, and refresh tuning uses code defaults.

## Docker

The Compose setup mirrors the simple server workflow used by `FreeAIchat-2api`: it reads `.env`, uses host networking, and persists data in `./data`.

```bash
mkdir -p data
cp .env.example .env
vim .env
docker compose up -d --build
docker compose logs -f qwen2api-rs
curl -s http://127.0.0.1:7860/healthz
```

Because `network_mode: host` is used, `PORT` is the host port directly.

## Binary

```bash
cargo run
```

or run a built binary with the same `.env` file in the working directory.

## API Examples

OpenAI compatible:

```bash
curl http://127.0.0.1:7860/v1/chat/completions \
  -H "Authorization: Bearer sk-qwen2api-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Anthropic compatible:

```bash
curl http://127.0.0.1:7860/v1/messages \
  -H "x-api-key: sk-qwen2api-change-me" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-5-sonnet","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'
```

Gemini compatible:

```bash
curl "http://127.0.0.1:7860/v1beta/models/gemini-2.5-pro:generateContent" \
  -H "x-goog-api-key: sk-qwen2api-change-me" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'
```

## Main Files

```text
src/main.rs             routes and lifecycle
src/config.rs           small public env surface plus internal defaults
src/account/            Qwen account pool
src/upstream/           Qwen HTTP/SSE client, payloads, executor, chat_id prewarm
src/request/            protocol request normalization
src/execution/          stream normalization and protocol output translation
src/toolcall/           prompt-injected tool calls and parser
src/context/            files, attachments, OSS upload
src/media.rs            image/video task queue and local backups
src/api/                OpenAI, Anthropic, Gemini, files, media, probes
```

## Notes

Use this project only for self-hosted experimentation and comply with the upstream service terms. Qwen Web behavior can change without notice.
