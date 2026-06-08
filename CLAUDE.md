# qwen2api — Project AI Context

Qwen Web gateway exposed as OpenAI/Anthropic compatible APIs. **The active backend is Python FastAPI**, aligned with the `FreeAIchat-2api` stack: `fastapi`, `uvicorn`, `curl_cffi`, `pydantic-settings`, and `python-dotenv`.
The gateway supports normal chat plus thinking/text streaming only. External action invocation protocols are intentionally not implemented. See `.env.example` for supported deployment variables.

## 開發 / 執行注意
- Install dependencies with a mainland mirror: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`.
- Local config is read from `.env`; authentication uses `API_KEY`.
- Qwen accounts live in `data/accounts.json` or `QWEN_ACCOUNTS_JSON`.
- Local run: `uvicorn main:app --host 0.0.0.0 --port 7860`.

## 部署（Docker，正式環境）
- Dockerfile uses Python 3.12 slim, Aliyun Debian mirrors, and Tsinghua PyPI.
- `docker-compose.yml` uses host networking and exposes `PORT=7860` directly.
- Persistent data is `data/accounts.json`. Configure outbound proxy with `UPSTREAM_PROXY` or standard `HTTP_PROXY/HTTPS_PROXY`.
- Deployment instructions live in `README.md`.

## 🔁 [流程] 每次開發完成後的重新部署
After code changes pass Python syntax/import checks and a health check:
```bash
docker compose up -d --build
docker compose logs -f --tail=30
curl -s http://127.0.0.1:7860/healthz
```
Data under `data/` is persistent across rebuilds.
