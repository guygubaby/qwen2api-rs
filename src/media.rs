//! 媒體生成子系統：本地保存 + 任務佇列 + 帶重試的生成（圖片 t2i / 影片 t2v 共用）。
//!
//! 背景：上游 t2i/t2v 都是同步 SSE，無可輪詢 task API；且影片有「每日額度上限」。
//! 故本模組提供：
//! - `generate_with_retry`：應用層重試，搭配 executor 的限流分類，輪換到有額度的帳號。
//! - `MediaStore`：SQLite（data/media.db）持久化任務，本地落盤生成的圖/影片（data/generated_media/）。
//! - `MediaQueue`：背景 worker（受 Semaphore 限制並發），消費佇列任務並下載存檔。
//!
//! 對外 API（/v1/images、/v1/videos）仍回 Qwen CDN 原始 URL，僅「額外」存本地備份（使用者決策）。

use crate::request::StandardRequest;
use crate::state::AppState;
use crate::upstream::ImageOptions;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, Semaphore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

impl MediaKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Video => "video",
        }
    }
    pub fn chat_type(&self) -> &'static str {
        match self {
            MediaKind::Image => "t2i",
            MediaKind::Video => "t2v",
        }
    }
    pub fn surface(&self) -> &'static str {
        match self {
            MediaKind::Image => "images",
            MediaKind::Video => "videos",
        }
    }
    pub fn parse(s: &str) -> MediaKind {
        if s.eq_ignore_ascii_case("video") || s == "t2v" {
            MediaKind::Video
        } else {
            MediaKind::Image
        }
    }
    fn default_ext(&self) -> &'static str {
        match self {
            MediaKind::Image => "png",
            MediaKind::Video => "mp4",
        }
    }
}

/// 從上游回應文字抓圖片/影片 URL（合併原 images.rs / videos.rs 的抓取邏輯）。
pub fn scrape_urls(text: &str, kind: MediaKind) -> Vec<String> {
    use once_cell::sync::Lazy;
    use regex::Regex;
    static RE_URL: Lazy<Regex> = Lazy::new(|| Regex::new(r#"https?://[^\s"'<>)\]]+"#).unwrap());
    let mut seen: Vec<String> = Vec::new();
    for m in RE_URL.find_iter(text) {
        let u = m.as_str().trim_end_matches(['.', ',', ')', ']']).to_string();
        let low = u.to_lowercase();
        let hit = match kind {
            MediaKind::Image => {
                low.contains("cdn.qwenlm.ai")
                    || low.contains("alicdn.com")
                    || low.contains("/t2i/")
                    || [".png", ".jpg", ".jpeg", ".webp", ".gif"].iter().any(|e| low.contains(e))
            }
            MediaKind::Video => {
                low.contains("/t2v/")
                    || low.contains("cdn.qwenlm.ai")
                    || [".mp4", ".webm", ".mov"].iter().any(|e| low.contains(e))
            }
        };
        if hit && !seen.contains(&u) {
            seen.push(u);
        }
    }
    seen
}

/// 一次生成的結果。
pub struct GenOutcome {
    pub urls: Vec<String>,
    pub attempts: u32,
    pub error: Option<String>,
    /// 本次過程中被新標記為「t2v 無權限」的帳號數量（供面板統計）。
    pub newly_marked_no_t2v: usize,
}

/// 帶重試的單次生成：迴圈呼叫 collect_completion，命中 URL 即回。
///
/// **智慧跳過**（僅影片 t2v）：
/// - 起始 exclude = 持久化的 no_t2v 集合 ∪ session 內已驗證無權限的帳號。
/// - 若某次「成功但無 URL」且有 used_email → 該帳號無 t2v 權限，加入 no_t2v 集（持久化）並排除後續嘗試。
/// - 若所有可用帳號都被排除 → 提早結束並回清楚錯誤。
///
/// 圖片 t2i 不啟用智慧跳過（實測幾乎所有帳號都能生成）。
pub async fn generate_with_retry(
    state: &AppState,
    prompt: &str,
    kind: MediaKind,
    options: ImageOptions,
    max_attempts: u32,
    caller: Option<String>,
) -> GenOutcome {
    let resolved = state.resolve_model(&state.settings.default_model).await;
    let response_model = match kind {
        MediaKind::Image => "qwen-image",
        MediaKind::Video => "qwen-video",
    };
    let attempts = max_attempts.max(1);
    let mut last_error: Option<String> = None;

    // 智慧跳過：載入持久化的 no_t2v 集合作為初始 exclude（僅影片）。
    let mut exclude: HashSet<String> = if kind == MediaKind::Video {
        state.no_t2v.get().await
    } else {
        HashSet::new()
    };
    let initial_skipped = exclude.len();
    let mut newly_marked: HashSet<String> = HashSet::new();

    for attempt in 1..=attempts {
        if kind == MediaKind::Video {
            let pool_status = state.pool.status().await;
            let valid = pool_status.get("valid").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            if valid > 0 && exclude.len() >= valid {
                tracing::warn!(
                    "[media] video attempt {}: skip set covers all {} valid accounts; stopping early",
                    attempt, valid
                );
                last_error = Some("no remaining accounts with t2v access to try".to_string());
                break;
            }
        }
        let std = StandardRequest {
            response_model: response_model.to_string(),
            resolved_model: resolved.clone(),
            prompt: prompt.to_string(),
            stream: false,
            thinking_enabled: Some(false),
            force_thinking: false,
            enable_search: false,
            chat_type: kind.chat_type().to_string(),
            tools: vec![],
            tool_names: vec![],
            surface: kind.surface().to_string(),
            image_options: Some(options.clone()),
            max_tokens: None,
            client_profile: "generic".to_string(),
            files: vec![],
            bound_account: None,
            caller: caller.clone(),
            exclude_accounts: exclude.clone(),
        };
        let r = crate::execution::collect_completion(state.clone(), std, HashMap::new()).await;

        if let Some(e) = &r.error {
            last_error = Some(e.clone());
            if e.contains("no available Qwen accounts") || e.contains("no remaining accounts") {
                tracing::warn!(
                    "[media] {} attempt {}: no available accounts (exclude={}); stopping early",
                    kind.as_str(), attempt, exclude.len()
                );
                break;
            }
            tracing::warn!("[media] {} attempt {}/{} failed: {}", kind.as_str(), attempt, attempts, e);
            continue;
        }
        let urls = scrape_urls(&r.content, kind);
        if !urls.is_empty() {
            // 把本輪新發現的無 t2v 帳號一次性持久化（影片才有）
            if kind == MediaKind::Video && !newly_marked.is_empty() {
                let to_add = newly_marked.clone();
                state.no_t2v.update(|s| { for em in to_add { s.insert(em); } }).await;
            }
            return GenOutcome { urls, attempts: attempt, error: None, newly_marked_no_t2v: newly_marked.len() };
        }
        // 成功但無 URL：影片＝該帳號無 t2v 權限；圖片＝視為一般可重試（少見）
        if kind == MediaKind::Video {
            if let Some(em) = r.email {
                if !exclude.contains(&em) {
                    tracing::info!("[media] marking account without t2v access email={em}");
                    exclude.insert(em.clone());
                    newly_marked.insert(em);
                }
            }
        }
        last_error = Some(format!(
            "upstream did not return a {} URL",
            if kind == MediaKind::Video { "video" } else { "image" }
        ));
        tracing::warn!("[media] {} attempt {}/{} returned no URL (skipped {} accounts without access); retrying",
            kind.as_str(), attempt, attempts, exclude.len());
    }
    // 失敗也要把已標記的持久化（不讓下次又踩同一批帳號）
    if kind == MediaKind::Video && !newly_marked.is_empty() {
        let to_add = newly_marked.clone();
        state.no_t2v.update(|s| { for em in to_add { s.insert(em); } }).await;
    }
    // 失敗錯誤訊息要清楚
    let final_err = if kind == MediaKind::Video {
        Some(format!(
            "video generation failed: no URL after {} attempts (newly skipped {} accounts without t2v access; previously skipped {}). Upstream returned an empty result, which usually means the account does not have t2v access. Last error: {}",
            attempts,
            newly_marked.len(),
            initial_skipped,
            last_error.unwrap_or_else(|| "none".into())
        ))
    } else {
        last_error
    };
    GenOutcome { urls: vec![], attempts, error: final_err, newly_marked_no_t2v: newly_marked.len() }
}

/// 下載媒體 URL 到本地，回傳檔名（失敗回 None）。經 QwenClient 的 reqwest（含出口代理）。
pub async fn download_to_local(
    client: &reqwest::Client,
    media_dir: &Path,
    url: &str,
    kind: MediaKind,
) -> Option<String> {
    let resp = client.get(url).timeout(Duration::from_secs(180)).send().await.ok()?;
    if !resp.status().is_success() {
        tracing::warn!("[media] download failed HTTP {} url={}", resp.status(), &url[..url.len().min(80)]);
        return None;
    }
    let bytes = resp.bytes().await.ok()?;
    if bytes.is_empty() {
        return None;
    }
    let ext = guess_ext(url, kind);
    let filename = format!("{}.{}", crate::util::short_id(28), ext);
    if tokio::fs::create_dir_all(media_dir).await.is_err() {
        return None;
    }
    let path = media_dir.join(&filename);
    if tokio::fs::write(&path, &bytes).await.is_err() {
        return None;
    }
    tracing::info!("[media] saved local copy {} ({} bytes)", filename, bytes.len());
    Some(filename)
}

fn guess_ext(url: &str, kind: MediaKind) -> String {
    let low = url.split('?').next().unwrap_or(url).to_lowercase();
    for e in ["png", "jpg", "jpeg", "webp", "gif", "mp4", "webm", "mov"] {
        if low.ends_with(&format!(".{e}")) {
            return e.to_string();
        }
    }
    kind.default_ext().to_string()
}

// ===================== 持久化（data/media.db）=====================

fn open_conn(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(conn)
}

/// 任務列（供前端渲染）。
#[derive(Debug, Clone)]
pub struct MediaTask {
    pub id: i64,
    pub kind: String,
    pub prompt: String,
    pub params: String,
    pub status: String,
    pub caller: Option<String>,
}

pub struct MediaStore {
    db_path: PathBuf,
    pub media_dir: PathBuf,
}

impl MediaStore {
    pub fn new(db_path: impl AsRef<Path>, media_dir: impl AsRef<Path>) -> Arc<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        let media_dir = media_dir.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::create_dir_all(&media_dir);
        if let Err(e) = init_schema(&db_path) {
            tracing::error!("[media] database initialization failed: {e}");
        }
        Arc::new(MediaStore { db_path, media_dir })
    }

    /// 建立排隊中的任務，回傳 task id。
    pub async fn create_task(&self, kind: MediaKind, prompt: &str, params: Value, caller: Option<String>) -> Option<i64> {
        let path = self.db_path.clone();
        let (k, p, pr, c) = (kind.as_str().to_string(), prompt.to_string(), params.to_string(), caller);
        let now = crate::util::now_millis();
        spawn_db(move || {
            let conn = open_conn(&path)?;
            conn.execute(
                "INSERT INTO media_tasks (ts_created,kind,prompt,params,status,attempts,caller)
                 VALUES (?1,?2,?3,?4,'queued',0,?5)",
                params![now, k, p, pr, c],
            )?;
            Ok(conn.last_insert_rowid())
        })
        .await
    }

    /// 取出排隊中的任務（標記為 running），回傳其 id。單一 dispatcher 呼叫，無競態。
    pub async fn take_queued(&self, limit: usize) -> Vec<i64> {
        let path = self.db_path.clone();
        spawn_db(move || {
            let conn = open_conn(&path)?;
            let ids: Vec<i64> = {
                let mut stmt = conn.prepare("SELECT id FROM media_tasks WHERE status='queued' ORDER BY id LIMIT ?1")?;
                let rows = stmt.query_map(params![limit as i64], |r| r.get::<_, i64>(0))?;
                rows.collect::<rusqlite::Result<Vec<i64>>>()?
            };
            for id in &ids {
                conn.execute("UPDATE media_tasks SET status='running' WHERE id=?1", params![id])?;
            }
            Ok(ids)
        })
        .await
        .unwrap_or_default()
    }

    /// 啟動時把殘留的 running（前次中斷）重新排隊。
    pub async fn requeue_stale(&self) {
        let path = self.db_path.clone();
        let _: Option<usize> = spawn_db(move || {
            let conn = open_conn(&path)?;
            let n = conn.execute("UPDATE media_tasks SET status='queued' WHERE status='running'", [])?;
            if n > 0 {
                tracing::info!("[media] requeued {n} interrupted tasks");
            }
            Ok(n)
        })
        .await;
    }

    pub async fn get(&self, id: i64) -> Option<MediaTask> {
        let path = self.db_path.clone();
        spawn_db(move || {
            let conn = open_conn(&path)?;
            conn.query_row(
                "SELECT id,kind,prompt,params,status,caller FROM media_tasks WHERE id=?1",
                params![id],
                |r| {
                    Ok(MediaTask {
                        id: r.get(0)?,
                        kind: r.get(1)?,
                        prompt: r.get(2)?,
                        params: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        status: r.get(4)?,
                        caller: r.get(5)?,
                    })
                },
            )
        })
        .await
    }

    pub async fn complete(&self, id: i64, results: Value, attempts: u32) {
        let path = self.db_path.clone();
        let now = crate::util::now_millis();
        let res = results.to_string();
        let _: Option<usize> = spawn_db(move || {
            let conn = open_conn(&path)?;
            Ok(conn.execute(
                "UPDATE media_tasks SET status='done', results=?2, attempts=?3, ts_done=?4, error=NULL WHERE id=?1",
                params![id, res, attempts as i64, now],
            )?)
        })
        .await;
    }

    pub async fn fail(&self, id: i64, error: &str, attempts: u32) {
        let path = self.db_path.clone();
        let now = crate::util::now_millis();
        let err = error.to_string();
        let _: Option<usize> = spawn_db(move || {
            let conn = open_conn(&path)?;
            Ok(conn.execute(
                "UPDATE media_tasks SET status='failed', error=?2, attempts=?3, ts_done=?4 WHERE id=?1",
                params![id, err, attempts as i64, now],
            )?)
        })
        .await;
    }

    /// 直接插入一筆已完成任務（給同步 API 的本地備份用，狀態 done 不會被 worker 重生）。
    pub async fn insert_done(&self, kind: MediaKind, prompt: &str, params: Value, results: Value, caller: Option<String>) -> Option<i64> {
        let path = self.db_path.clone();
        let (k, p, pr, res, c) = (kind.as_str().to_string(), prompt.to_string(), params.to_string(), results.to_string(), caller);
        let now = crate::util::now_millis();
        spawn_db(move || {
            let conn = open_conn(&path)?;
            conn.execute(
                "INSERT INTO media_tasks (ts_created,ts_done,kind,prompt,params,status,attempts,results,caller)
                 VALUES (?1,?1,?2,?3,?4,'done',1,?5,?6)",
                params![now, k, p, pr, res, c],
            )?;
            Ok(conn.last_insert_rowid())
        })
        .await
    }

    /// 下載一批 URL 到本地，回傳 results JSON 陣列（cdn_url/filename/local_url）。
    pub async fn backup_urls(&self, client: &reqwest::Client, urls: &[String], kind: MediaKind) -> Value {
        let mut out = Vec::new();
        for url in urls {
            let filename = download_to_local(client, &self.media_dir, url, kind).await;
            let local_url = filename.as_ref().map(|f| format!("/media/{f}"));
            out.push(json!({ "cdn_url": url, "filename": filename, "local_url": local_url }));
        }
        json!(out)
    }

    /// 列出最近任務（含結果），供前端畫廊渲染。
    pub async fn list(&self, kind: Option<MediaKind>, limit: i64) -> Value {
        let path = self.db_path.clone();
        let kind_filter = kind.map(|k| k.as_str().to_string());
        spawn_db(move || {
            let conn = open_conn(&path)?;
            let sql = if kind_filter.is_some() {
                "SELECT id,ts_created,ts_done,kind,prompt,params,status,attempts,error,results,caller
                 FROM media_tasks WHERE kind=?2 ORDER BY id DESC LIMIT ?1"
            } else {
                "SELECT id,ts_created,ts_done,kind,prompt,params,status,attempts,error,results,caller
                 FROM media_tasks ORDER BY id DESC LIMIT ?1"
            };
            let mut stmt = conn.prepare(sql)?;
            let map = |r: &rusqlite::Row| -> rusqlite::Result<Value> {
                let results_str: Option<String> = r.get(9)?;
                let results: Value = results_str
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(Value::Array(vec![]));
                let params_str: Option<String> = r.get(5)?;
                let prm: Value = params_str
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(Value::Null);
                Ok(json!({
                    "id": r.get::<_, i64>(0)?,
                    "ts_created": r.get::<_, i64>(1)?,
                    "ts_done": r.get::<_, Option<i64>>(2)?,
                    "kind": r.get::<_, String>(3)?,
                    "prompt": r.get::<_, String>(4)?,
                    "params": prm,
                    "status": r.get::<_, String>(6)?,
                    "attempts": r.get::<_, i64>(7)?,
                    "error": r.get::<_, Option<String>>(8)?,
                    "results": results,
                    "caller": r.get::<_, Option<String>>(10)?,
                }))
            };
            let rows: Vec<Value> = {
                let mut collected = Vec::new();
                let mut q = if kind_filter.is_some() {
                    stmt.query(params![limit.clamp(1, 500), kind_filter.unwrap()])?
                } else {
                    stmt.query(params![limit.clamp(1, 500)])?
                };
                while let Some(row) = q.next()? {
                    collected.push(map(row)?);
                }
                collected
            };
            Ok(json!({ "tasks": rows }))
        })
        .await
        .unwrap_or_else(|| json!({ "tasks": [] }))
    }
}

fn init_schema(path: &Path) -> rusqlite::Result<()> {
    let conn = open_conn(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS media_tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_created  INTEGER NOT NULL,
            ts_done     INTEGER,
            kind        TEXT NOT NULL,
            prompt      TEXT NOT NULL,
            params      TEXT,
            status      TEXT NOT NULL DEFAULT 'queued',
            attempts    INTEGER NOT NULL DEFAULT 0,
            error       TEXT,
            results     TEXT,
            caller      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_media_status ON media_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_media_created ON media_tasks(ts_created);",
    )?;
    Ok(())
}

/// 在 blocking 執行緒池跑同步 DB 操作；JoinError/SQL error 皆回 None 並記 log。
async fn spawn_db<T, F>(f: F) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> rusqlite::Result<T> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => Some(v),
        Ok(Err(e)) => {
            tracing::error!("[media] DB operation failed: {e}");
            None
        }
        Err(e) => {
            tracing::error!("[media] DB task panicked: {e}");
            None
        }
    }
}

// ===================== 任務佇列 + 背景 worker =====================

pub struct MediaQueue {
    pub store: Arc<MediaStore>,
    notify: Arc<Notify>,
    concurrency: usize,
    attempts: u32,
}

impl MediaQueue {
    pub fn new(store: Arc<MediaStore>, concurrency: usize, attempts: u32) -> Arc<Self> {
        Arc::new(MediaQueue {
            store,
            notify: Arc::new(Notify::new()),
            concurrency: concurrency.max(1),
            attempts: attempts.max(1),
        })
    }

    /// 提交任務（排隊），回傳 task id 並喚醒 worker。
    pub async fn submit(&self, kind: MediaKind, prompt: &str, params: Value, caller: Option<String>) -> Option<i64> {
        let id = self.store.create_task(kind, prompt, params, caller).await;
        if id.is_some() {
            self.notify.notify_one();
        }
        id
    }

    /// 啟動背景 dispatcher（在 main 內以 state clone 呼叫，避免 Arc 循環）。
    pub fn start(self: Arc<Self>, state: AppState) {
        tokio::spawn(async move {
            self.store.requeue_stale().await;
            let sem = Arc::new(Semaphore::new(self.concurrency));
            loop {
                let free = sem.available_permits();
                if free == 0 {
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    continue;
                }
                let ids = self.store.take_queued(free).await;
                if ids.is_empty() {
                    tokio::select! {
                        _ = self.notify.notified() => {}
                        _ = tokio::time::sleep(Duration::from_secs(3)) => {}
                    }
                    continue;
                }
                for id in ids {
                    let permit = match sem.clone().acquire_owned().await {
                        Ok(p) => p,
                        Err(_) => break,
                    };
                    let st = state.clone();
                    let store = self.store.clone();
                    let attempts = self.attempts;
                    tokio::spawn(async move {
                        process_task(st, store, id, attempts).await;
                        drop(permit);
                    });
                }
            }
        });
    }
}

/// 處理單一任務：生成（帶重試）→ 下載存本地 → 更新狀態。
async fn process_task(state: AppState, store: Arc<MediaStore>, id: i64, attempts: u32) {
    let task = match store.get(id).await {
        Some(t) => t,
        None => return,
    };
    let kind = MediaKind::parse(&task.kind);
    let prm: Value = serde_json::from_str(&task.params).unwrap_or(Value::Null);
    let options = params_to_image_options(&prm, kind);
    // 圖片支援 n 張；影片固定 1。
    let n = if kind == MediaKind::Image {
        prm.get("n").and_then(|v| v.as_u64()).unwrap_or(1).clamp(1, 4) as u32
    } else {
        1
    };

    let client = state.client.client();
    let mut results: Vec<Value> = Vec::new();
    let mut total_attempts = 0u32;
    let mut last_err: Option<String> = None;

    for _ in 0..n {
        let out = generate_with_retry(&state, &task.prompt, kind, options.clone(), attempts, task.caller.clone()).await;
        total_attempts += out.attempts;
        if out.urls.is_empty() {
            last_err = out.error;
            continue;
        }
        for url in out.urls {
            let filename = download_to_local(&client, &store.media_dir, &url, kind).await;
            let local_url = filename.as_ref().map(|f| format!("/media/{f}"));
            results.push(json!({ "cdn_url": url, "filename": filename, "local_url": local_url }));
        }
    }

    if results.is_empty() {
        store.fail(id, &last_err.unwrap_or_else(|| "generation failed".into()), total_attempts).await;
    } else {
        store.complete(id, json!(results), total_attempts).await;
    }
}

/// 從前端參數（ratio/size/width/height）建 ImageOptions。
fn params_to_image_options(prm: &Value, kind: MediaKind) -> ImageOptions {
    let ratio = prm
        .get("ratio")
        .or_else(|| prm.get("aspect_ratio"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| Some(if kind == MediaKind::Video { "16:9".into() } else { "1:1".into() }));
    ImageOptions {
        size: prm.get("size").and_then(|v| v.as_str()).map(|s| s.to_string()),
        ratio,
        width: prm.get("width").and_then(|v| v.as_i64()),
        height: prm.get("height").and_then(|v| v.as_i64()),
    }
}
