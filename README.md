# qwen2api Python

FastAPI gateway that exposes Qwen Web as OpenAI and Anthropic compatible APIs. The implementation follows the current qwen2api flow, but uses the same Python stack style as `FreeAIchat-2api`: `fastapi`, `uvicorn`, `curl_cffi`, `pydantic-settings`, and `python-dotenv`.

The Python implementation only forwards normal chat content and thinking/text streaming.

## Endpoints

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /chat/completions`
- `POST /v1/messages`
- `POST /messages`
- `POST /anthropic/v1/messages`

## Model Modes

`/v1/models` returns both base models and `-thinking` aliases. For example:

- `qwen3.7-plus`
- `qwen3.7-plus-thinking`
- `qwen3.7-max`
- `qwen3.7-max-thinking`

When a request uses a model ending in `-thinking`, the gateway enables Qwen thinking and sends the base model upstream. For example, `qwen3.7-plus-thinking` is sent to Qwen as `qwen3.7-plus`.

Thinking can also be enabled explicitly with:

```json
{"thinking_enabled": true}
```

Without the suffix or explicit flag, thinking is disabled by default.

## Configuration

```bash
cp .env.example .env
mkdir -p data
```

Minimal `.env`:

```env
API_KEY=sk-qwen2api-change-me
PORT=7860
DATA_DIR=./data
DEFAULT_MODEL=qwen3.7-plus
```

Provide Qwen accounts through `data/accounts.json`:

```json
[
  {
    "email": "user@example.com",
    "token": "token-from-chat.qwen.ai-localStorage"
  }
]
```

Or pass the same JSON array through `QWEN_ACCOUNTS_JSON`.

## Docker

The Dockerfile uses domestic mirrors for mainland China deployments:

- Debian apt: `mirrors.aliyun.com`
- PyPI: `https://pypi.tuna.tsinghua.edu.cn/simple`

```bash
docker compose up -d --build
docker compose logs -f qwen2api
curl -s http://127.0.0.1:7860/healthz
```

`network_mode: host` is used, so `PORT` maps directly to the host.

## Local Run

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Examples

OpenAI compatible:

```bash
curl http://127.0.0.1:7860/v1/chat/completions \
  -H "Authorization: Bearer sk-qwen2api-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.7-plus","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Anthropic compatible:

```bash
curl http://127.0.0.1:7860/v1/messages \
  -H "x-api-key: sk-qwen2api-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.7-plus","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```
