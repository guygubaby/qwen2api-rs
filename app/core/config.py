import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "qwen2api-python"
    APP_VERSION: str = "3.0.0"
    DESCRIPTION: str = "Qwen Web to OpenAI/Anthropic compatible API gateway."

    API_KEY: str = "sk-qwen2api-change-me"
    PORT: int = 7860
    DATA_DIR: str = "./data"
    ACCOUNTS_FILE: str | None = None
    QWEN_ACCOUNTS_JSON: str | None = None
    UPSTREAM_PROXY: str | None = None
    DEFAULT_MODEL: str = "qwen3.7-plus"
    LOG_LEVEL: str = "info"

    QWEN_BASE_URL: str = "https://chat.qwen.ai"
    QWEN_REQUEST_TIMEOUT_SECONDS: int = 300
    CHAT_DELETE_ON_CLOSE: bool = True
    MAX_INFLIGHT_PER_ACCOUNT: int = 1
    ACCOUNT_MIN_INTERVAL_MS: int = 3000
    EMPTY_RESPONSE_RETRY_ATTEMPTS: int = 1
    EMPTY_RESPONSE_COOLDOWN_MS: int = 60000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def accounts_path(self) -> Path:
        if self.ACCOUNTS_FILE:
            return Path(self.ACCOUNTS_FILE)
        return Path(self.DATA_DIR) / "accounts.json"

    def load_accounts(self) -> list[dict[str, Any]]:
        if self.QWEN_ACCOUNTS_JSON:
            data = json.loads(self.QWEN_ACCOUNTS_JSON)
        else:
            path = self.accounts_path
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("accounts must be a JSON array")
        return [item for item in data if isinstance(item, dict) and item.get("token")]


settings = Settings()
