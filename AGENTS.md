# qwen2api — 專案 AI 上下文

把 Qwen Web 偽裝成 OpenAI/Anthropic 相容介面的網關。**後端已切換為 Python FastAPI**，技術棧對齊 `FreeAIchat-2api`：`fastapi`、`uvicorn`、`curl_cffi`、`pydantic-settings`、`python-dotenv`。
目前只保留普通聊天與 thinking/text 串流，不實作外部動作調用協議。所有可調 env 變數的權威清單在 `.env.example`。

## 開發 / 執行注意
- 依賴安裝優先用國內源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- 本地配置從 `.env` 讀取；鑑權使用 `API_KEY`，Qwen 帳號放 `data/accounts.json` 或 `QWEN_ACCOUNTS_JSON`。
- 本地啟動：`uvicorn main:app --host 0.0.0.0 --port 7860`。
- 健康檢查：`curl -s http://127.0.0.1:7860/healthz`。

## 部署（Docker，正式環境）
- Dockerfile 已切到 Python 3.12 slim，apt 使用阿里源，pip 使用清華源。
- `docker-compose.yml` 使用 host network，`PORT=7860` 直接對外。
- 持久化資料只需要 `data/accounts.json`；如果需要代理，通過 `.env` 配置 `UPSTREAM_PROXY` 或標準 `HTTP_PROXY/HTTPS_PROXY`。
- 部署方式見 `README.md`。

## 🔁 [流程] 每次開發完成後的重新部署
開發改完、Python 語法與健康檢查驗證 OK 後：
```bash
docker compose up -d --build
docker compose logs -f --tail=30
curl -s http://127.0.0.1:7860/healthz
```
資料在 `data/` 持久化，重部署不丟。
