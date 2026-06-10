import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.prompt import build_prompt
from app.providers.qwen import AccountPool, QwenClient, build_payload, parse_sse_line


logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("qwen2api")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, description=settings.DESCRIPTION)
pool = AccountPool()
client = QwenClient()
THINKING_MODEL_SUFFIX = "-thinking"
EMPTY_RESPONSE_MESSAGE = "Qwen upstream returned empty streams after retries; the account is likely rate limited."


class UpstreamEmptyResponseError(RuntimeError):
    pass


async def verify_api_key(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> None:
    token = x_api_key
    if not token and authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    if not token:
        raise HTTPException(status_code=401, detail="missing API key")
    if token != settings.API_KEY:
        raise HTTPException(status_code=403, detail="invalid API key")


def resolve_model(model: str | None) -> str:
    return str(model or settings.DEFAULT_MODEL)


def upstream_model(model: str | None) -> str:
    resolved = resolve_model(model)
    if resolved.endswith(THINKING_MODEL_SUFFIX):
        return resolved[: -len(THINKING_MODEL_SUFFIX)]
    return resolved


def model_enables_thinking(model: str | None) -> bool:
    return resolve_model(model).endswith(THINKING_MODEL_SUFFIX)


def request_enables_thinking(body: dict[str, Any]) -> bool:
    if model_enables_thinking(body.get("model")):
        return True
    if isinstance(body.get("thinking"), dict):
        return True
    return bool(body.get("thinking_enabled"))


def model_item(model: str) -> dict[str, str]:
    return {"id": model, "object": "model", "owned_by": "qwen"}


def model_items_with_thinking_variants(models: list[str]) -> list[dict[str, str]]:
    data = []
    seen = set()
    for model in models:
        base = upstream_model(model)
        for item in (base, f"{base}{THINKING_MODEL_SUFFIX}"):
            if item not in seen:
                seen.add(item)
                data.append(model_item(item))
    return data


def openai_sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def anthropic_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def qwen_events(body: dict[str, Any]) -> AsyncGenerator[tuple[str, str], None]:
    model = upstream_model(body.get("model"))
    prompt = build_prompt(body)
    thinking_enabled = request_enables_thinking(body)

    request_id = uuid.uuid4().hex[:8]
    max_attempts = max(1, settings.EMPTY_RESPONSE_RETRY_ATTEMPTS + 1)

    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        content_chars = 0
        reasoning_chars = 0
        delta_count = 0
        retry_empty = False
        cooldown_ms: int | None = None
        account = await pool.acquire()
        chat_id: str | None = None
        reasoning_seen = ""
        logger.debug(
            "[qwen-request:%s] start attempt=%d/%d model=%s thinking=%s account=%s prompt_chars=%d",
            request_id,
            attempt,
            max_attempts,
            model,
            thinking_enabled,
            account.email,
            len(prompt),
        )
        try:
            chat_id = await client.create_chat(account.token, model, "t2t")
            logger.debug("[qwen-request:%s] chat_created attempt=%d chat_id=%s", request_id, attempt, chat_id)
            payload = build_payload(
                chat_id=chat_id,
                model=model,
                prompt=prompt,
                thinking_enabled=thinking_enabled,
                enable_search=bool(body.get("enable_search") or body.get("web_search")),
            )
            async for line in client.stream_chat(account.token, chat_id, payload):
                for delta in parse_sse_line(line):
                    delta_count += 1
                    if delta.reasoning_cumulative:
                        if delta.reasoning_cumulative.startswith(reasoning_seen):
                            inc = delta.reasoning_cumulative[len(reasoning_seen):]
                        else:
                            inc = delta.reasoning_cumulative
                        reasoning_seen = delta.reasoning_cumulative
                        if inc:
                            reasoning_chars += len(inc)
                            yield ("reasoning", inc)
                    if delta.reasoning_incremental:
                        reasoning_chars += len(delta.reasoning_incremental)
                        yield ("reasoning", delta.reasoning_incremental)
                    if delta.content and delta.phase not in {"think", "thinking_summary"}:
                        content_chars += len(delta.content)
                        yield ("content", delta.content)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if content_chars == 0 and reasoning_chars == 0:
                cooldown_ms = settings.EMPTY_RESPONSE_COOLDOWN_MS
                retry_empty = attempt < max_attempts
                logger.warning(
                    "[qwen-request:%s] empty_content attempt=%d/%d retry=%s model=%s account=%s chat_id=%s deltas=%d elapsed_ms=%d cooldown_ms=%d",
                    request_id,
                    attempt,
                    max_attempts,
                    retry_empty,
                    model,
                    account.email,
                    chat_id,
                    delta_count,
                    elapsed_ms,
                    cooldown_ms,
                )
            else:
                logger.debug(
                    "[qwen-request:%s] finish attempt=%d/%d model=%s account=%s chat_id=%s deltas=%d content_chars=%d reasoning_chars=%d elapsed_ms=%d",
                    request_id,
                    attempt,
                    max_attempts,
                    model,
                    account.email,
                    chat_id,
                    delta_count,
                    content_chars,
                    reasoning_chars,
                    elapsed_ms,
                )
                yield ("done", "")
                return
        except Exception:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.exception(
                "[qwen-request:%s] failed attempt=%d/%d model=%s account=%s chat_id=%s deltas=%d content_chars=%d reasoning_chars=%d elapsed_ms=%d",
                request_id,
                attempt,
                max_attempts,
                model,
                account.email,
                chat_id,
                delta_count,
                content_chars,
                reasoning_chars,
                elapsed_ms,
            )
            raise
        finally:
            if chat_id is not None:
                await client.delete_chat(account.token, chat_id)
            await pool.release(account, cooldown_ms=cooldown_ms)

        if not retry_empty:
            break

    logger.error("[qwen-request:%s] exhausted_empty_retries model=%s attempts=%d", request_id, model, max_attempts)
    raise UpstreamEmptyResponseError(EMPTY_RESPONSE_MESSAGE)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "accounts": pool.count(), "stack": "fastapi"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def models() -> dict[str, Any]:
    try:
        account = await pool.acquire()
        upstream = await client.list_models(account.token)
        models = [str(item.get("id") or item.get("name")) for item in upstream if item.get("id") or item.get("name")]
        data = model_items_with_thinking_variants(models)
    except Exception:
        data = []
    if not data:
        data = model_items_with_thinking_variants([settings.DEFAULT_MODEL])
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
@app.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    model = resolve_model(body.get("model"))
    stream = bool(body.get("stream"))
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        async def generate() -> AsyncGenerator[str, None]:
            yield openai_sse({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            })
            try:
                async for kind, text in qwen_events(body):
                    if kind == "content":
                        yield openai_sse({
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        })
            except UpstreamEmptyResponseError as exc:
                logger.warning("[openai-stream:%s] upstream_empty_response %s", chat_id, exc)
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': {'message': str(exc), 'type': 'rate_limit_error'}}, ensure_ascii=False)}\n\n"
                )
                yield "data: [DONE]\n\n"
                return
            yield openai_sse({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    content = ""
    try:
        async for kind, text in qwen_events(body):
            if kind == "content":
                content += text
    except UpstreamEmptyResponseError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/messages", dependencies=[Depends(verify_api_key)])
@app.post("/messages", dependencies=[Depends(verify_api_key)])
@app.post("/anthropic/v1/messages", dependencies=[Depends(verify_api_key)])
async def anthropic_messages(request: Request) -> Any:
    body = await request.json()
    model = resolve_model(body.get("model"))
    stream = bool(body.get("stream"))
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    if stream:
        async def generate() -> AsyncGenerator[str, None]:
            yield anthropic_sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
            index = 0
            current: str | None = None
            try:
                async for kind, text in qwen_events(body):
                    if kind == "reasoning":
                        if current != "thinking":
                            if current is not None:
                                yield anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
                                index += 1
                            current = "thinking"
                            yield anthropic_sse("content_block_start", {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "thinking", "thinking": "", "signature": "qwen2api-python"},
                            })
                        yield anthropic_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "thinking_delta", "thinking": text},
                        })
                    elif kind == "content":
                        if current != "text":
                            if current is not None:
                                yield anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
                                index += 1
                            current = "text"
                            yield anthropic_sse("content_block_start", {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            })
                        yield anthropic_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text},
                        })
            except UpstreamEmptyResponseError as exc:
                logger.warning("[anthropic-stream:%s] upstream_empty_response %s", message_id, exc)
                yield anthropic_sse("error", {
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": str(exc)},
                })
                return
            if current is not None:
                yield anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
            yield anthropic_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            })
            yield anthropic_sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(generate(), media_type="text/event-stream")

    content = ""
    try:
        async for kind, text in qwen_events(body):
            if kind == "content":
                content += text
    except UpstreamEmptyResponseError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


@app.exception_handler(Exception)
async def exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("request failed")
    return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "server_error"}})
