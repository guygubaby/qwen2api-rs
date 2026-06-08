import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from curl_cffi.requests import AsyncSession

from app.core.config import settings


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class Account:
    email: str
    token: str


@dataclass
class QwenDelta:
    phase: str = "answer"
    content: str = ""
    reasoning_cumulative: str | None = None
    reasoning_incremental: str = ""
    status: str = ""
    usage: dict[str, Any] | None = None


class AccountPool:
    def __init__(self) -> None:
        self._accounts = [
            Account(email=str(item.get("email") or f"account-{idx}"), token=str(item["token"]))
            for idx, item in enumerate(settings.load_accounts())
        ]
        self._index = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Account:
        async with self._lock:
            if not self._accounts:
                raise RuntimeError("no Qwen accounts configured")
            account = self._accounts[self._index % len(self._accounts)]
            self._index += 1
            return account

    def count(self) -> int:
        return len(self._accounts)


class QwenClient:
    def __init__(self) -> None:
        proxies = None
        if settings.UPSTREAM_PROXY:
            proxies = {"http": settings.UPSTREAM_PROXY, "https": settings.UPSTREAM_PROXY}
        self.session = AsyncSession(
            timeout=settings.QWEN_REQUEST_TIMEOUT_SECONDS,
            proxies=proxies,
            headers={
                "User-Agent": UA,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": f"{settings.QWEN_BASE_URL}/",
                "Origin": settings.QWEN_BASE_URL,
            },
        )

    def _headers(self, token: str, accept: str = "application/json, text/plain, */*") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": accept,
            "Content-Type": "application/json",
        }

    async def create_chat(self, token: str, model: str, chat_type: str = "t2t") -> str:
        ts = int(time.time())
        body = {
            "title": f"api_{ts}",
            "models": [model],
            "chat_mode": "normal",
            "chat_type": chat_type,
            "timestamp": ts,
        }
        resp = await self.session.post(
            f"{settings.QWEN_BASE_URL}/api/v2/chats/new",
            headers=self._headers(token),
            json=body,
        )
        text = resp.text
        if resp.status_code != 200:
            raise RuntimeError(f"create_chat HTTP {resp.status_code}: {text[:200]}")
        data = resp.json()
        if data.get("success") is not True:
            raise RuntimeError(f"create_chat success=false: {text[:200]}")
        chat_id = (data.get("data") or {}).get("id")
        if not chat_id:
            raise RuntimeError("create_chat response missing data.id")
        return str(chat_id)

    async def delete_chat(self, token: str, chat_id: str) -> None:
        if not settings.CHAT_DELETE_ON_CLOSE:
            return
        try:
            await self.session.delete(
                f"{settings.QWEN_BASE_URL}/api/v2/chats/{chat_id}",
                headers=self._headers(token),
            )
        except Exception:
            pass

    async def list_models(self, token: str) -> list[dict[str, Any]]:
        try:
            resp = await self.session.get(
                f"{settings.QWEN_BASE_URL}/api/models",
                headers=self._headers(token),
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = data.get("data")
            return models if isinstance(models, list) else []
        except Exception:
            return []

    async def stream_chat(
        self,
        token: str,
        chat_id: str,
        payload: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        url = f"{settings.QWEN_BASE_URL}/api/v2/chat/completions?chat_id={chat_id}"
        async with self.session.stream(
            "POST",
            url,
            headers=self._headers(token, accept="text/event-stream"),
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.acontent()
                raise RuntimeError(f"stream HTTP {resp.status_code}: {body[:200]!r}")
            async for chunk in resp.aiter_lines():
                if not chunk:
                    continue
                if isinstance(chunk, bytes):
                    yield chunk.decode("utf-8", errors="replace")
                else:
                    yield chunk


def uuid4() -> str:
    return str(uuid.uuid4())


def build_payload(
    chat_id: str,
    model: str,
    prompt: str,
    thinking_enabled: bool | None,
    enable_search: bool = False,
    chat_type: str = "t2t",
) -> dict[str, Any]:
    enabled = True if thinking_enabled is None else bool(thinking_enabled)
    if chat_type != "t2t":
        enabled = False
    feature_config = {
        "thinking_enabled": enabled,
        "auto_thinking": enabled,
        "thinking_mode": "Auto" if enabled else "Disabled",
        "thinking_format": "summary",
        "output_schema": "phase",
        "research_mode": "normal",
        "code_interpreter": False,
        "auto_search": enable_search,
    }
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "fid": uuid4(),
                "parentId": None,
                "childrenIds": [uuid4()],
                "role": "user",
                "content": prompt,
                "user_action": "chat",
                "files": [],
                "timestamp": int(time.time()),
                "models": [model],
                "chat_type": chat_type,
                "feature_config": feature_config,
                "extra": {"meta": {"subChatType": chat_type}},
            }
        ],
        "model": model,
    }


def _first_text(values: list[Any]) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _reasoning_incremental(delta: dict[str, Any]) -> str:
    extra = delta.get("extra") if isinstance(delta.get("extra"), dict) else {}
    return _first_text(
        [
            delta.get("reasoning_content"),
            delta.get("reasoning"),
            delta.get("reasoning_text"),
            delta.get("thinking"),
            delta.get("thoughts"),
            extra.get("reasoning_content"),
            extra.get("reasoning"),
            extra.get("reasoning_text"),
            extra.get("thinking"),
            extra.get("thoughts"),
        ]
    )


def _reasoning_cumulative(delta: dict[str, Any]) -> str | None:
    extra = delta.get("extra")
    if not isinstance(extra, dict):
        return None
    summary = extra.get("summary_thought")
    if not isinstance(summary, dict):
        return None
    content = summary.get("content")
    if not isinstance(content, list):
        return None
    text = "\n\n".join(str(item) for item in content if isinstance(item, str))
    return text or None


def parse_sse_line(line: str | bytes) -> list[QwenDelta]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line.startswith("data:"):
        return []
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return []
    return [
        QwenDelta(
            phase=str(delta.get("phase") or "answer"),
            content=str(delta.get("content") or ""),
            reasoning_cumulative=_reasoning_cumulative(delta),
            reasoning_incremental=_reasoning_incremental(delta),
            status=str(delta.get("status") or ""),
            usage=obj.get("usage") if isinstance(obj.get("usage"), dict) else None,
        )
    ]
