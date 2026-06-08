//! Global settings loaded from environment variables and `.env`.
//!
//! Public configuration is intentionally small. Most operational tuning values use
//! code defaults so deployments only need to pass the credentials and paths that
//! differ from the defaults.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_str(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_optional(key: &str) -> Option<String> {
    env::var(key).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// Runtime settings. Internal tuning is intentionally defaulted in code; the
/// environment surface stays close to the reference API-only project.
#[derive(Debug, Clone)]
pub struct Settings {
    pub port: u16,
    pub api_key: String,

    // 並發 / 容災
    pub max_inflight_per_account: i64,
    pub max_retries: u32,
    /// auth_error 連續失敗門檻：到達此值才永久標 valid=false（避免單次 401 誤殺）。
    /// 預設 3；中間幾次仍保留 valid、繼續嘗試，由 mark_success 重置計數。
    pub auth_error_fail_threshold: u32,
    pub account_min_interval_ms: u64,
    pub request_jitter_min_ms: u64,
    pub request_jitter_max_ms: u64,
    pub rate_limit_base_cooldown: u64,
    pub rate_limit_max_cooldown: u64,

    // 上游 chat 生命週期
    pub chat_delete_retry_attempts: u32,
    pub chat_delete_retry_delay_ms: u64,
    pub chat_id_prewarm_target_per_account: usize,
    pub chat_id_prewarm_ttl_seconds: u64,
    /// 預熱池最多覆蓋幾個帳號（優化：帳號數可能上萬，避免對全部帳號建會話打爆上游）。
    pub chat_id_prewarm_max_accounts: usize,

    pub log_level: String,

    // 資料檔案路徑
    pub data_dir: PathBuf,
    pub accounts_file: PathBuf,
    /// Optional JSON array of accounts supplied directly through env.
    pub accounts_json: Option<String>,
    pub config_file: PathBuf,
    /// 請求統計 SQLite 檔（data/stats.db）。
    pub stats_file: PathBuf,
    /// 媒體任務 SQLite 檔（data/media.db）。
    pub media_db_file: PathBuf,
    /// 生成媒體本地保存目錄（data/generated_media）。
    pub media_dir: PathBuf,
    /// 媒體任務 worker 並發數。
    pub media_concurrency: usize,
    /// 媒體生成的應用層重試次數（輪換帳號找有額度者）。
    pub media_max_attempts: u32,
    /// t2v 已知無權限的帳號集（持久化）。
    pub no_t2v_file: PathBuf,

    // 上下文 / 附件
    pub context_inline_max_chars: usize,
    pub context_force_file_max_chars: usize,
    pub context_attachment_ttl_seconds: u64,
    pub context_upload_parse_timeout_seconds: u64,
    pub context_generated_dir: PathBuf,
    pub context_cache_file: PathBuf,
    pub uploaded_files_file: PathBuf,
    pub context_affinity_file: PathBuf,
    pub context_allowed_user_exts: String,

    /// chat_id 預熱池預設模型（動態：啟動後若抓到上游模型列表會覆蓋）
    pub default_model: String,

    /// 出口全局代理初始值（優先 UPSTREAM_PROXY，否則沿用 HTTP(S)_PROXY env）。
    /// 之後可在管理台即時覆蓋並持久化。
    pub upstream_proxy: Option<String>,

    /// Pillar 2：就緒帳號索引（Ready-Set）。false 回退舊 O(n) 掃描（kill-switch）。預設開。
    pub pool_ready_index: bool,
    /// Pillar 3：連線保活。每 N 秒對上游送一次輕量請求保溫一條連線；0=關閉（預設，風控敏感）。
    pub conn_keepalive_seconds: u64,

    /// Token refresh worker（解決「JWT 30 天 TTL → 16k 帳號集體過期」）。
    /// 每 INTERVAL_HOURS 跑一輪：解所有帳號 JWT exp，篩出 exp < now+AHEAD_DAYS 的，分批跑
    /// chat.qwen.ai signin 拿新 token；每帳號間 JITTER_MS 隨機停頓避風控。0=關閉 worker。
    pub token_refresh_interval_hours: u64,
    pub token_refresh_ahead_days: i64,
    pub token_refresh_batch_per_cycle: usize,
    pub token_refresh_jitter_min_ms: u64,
    pub token_refresh_jitter_max_ms: u64,
}

/// 依序讀取代理環境變數（含大小寫變體）。
fn read_proxy_env() -> Option<String> {
    for key in ["UPSTREAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"] {
        if let Ok(v) = env::var(key) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

impl Settings {
    pub fn from_env() -> Self {
        let data_dir = PathBuf::from(env_str("DATA_DIR", "./data"));
        let data_path = |default_name: &str| -> PathBuf { data_dir.join(default_name) };
        Settings {
            port: env_or("PORT", 7860u16),
            api_key: env_str("API_KEY", "change-me-now"),
            max_inflight_per_account: 2,
            max_retries: 3,
            auth_error_fail_threshold: 3,
            account_min_interval_ms: 3000,
            request_jitter_min_ms: 0,
            request_jitter_max_ms: 0,
            rate_limit_base_cooldown: 600,
            rate_limit_max_cooldown: 3600,
            chat_delete_retry_attempts: 3,
            chat_delete_retry_delay_ms: 500,
            chat_id_prewarm_target_per_account: 5,
            chat_id_prewarm_ttl_seconds: 120,
            chat_id_prewarm_max_accounts: 8,
            log_level: env_str("LOG_LEVEL", "info"),
            accounts_file: env_optional("ACCOUNTS_FILE").map(PathBuf::from).unwrap_or_else(|| data_path("accounts.json")),
            accounts_json: env_optional("QWEN_ACCOUNTS_JSON"),
            config_file: data_path("config.json"),
            stats_file: data_path("stats.db"),
            media_db_file: data_path("media.db"),
            media_dir: data_path("generated_media"),
            media_concurrency: 3,
            media_max_attempts: 10,
            no_t2v_file: data_path("no_t2v_accounts.json"),
            context_inline_max_chars: 4000,
            context_force_file_max_chars: 10000,
            context_attachment_ttl_seconds: 1800,
            context_upload_parse_timeout_seconds: 60,
            context_generated_dir: data_path("context_files"),
            context_cache_file: data_path("context_cache.json"),
            uploaded_files_file: data_path("uploaded_files.json"),
            context_affinity_file: data_path("session_affinity.json"),
            context_allowed_user_exts:
                "txt,md,json,log,xml,yaml,yml,csv,html,css,py,js,ts,java,c,cpp,cs,php,go,rb,sh,zsh,ps1,bat,cmd,pdf,doc,docx,ppt,pptx,xls,xlsx,png,jpg,jpeg,webp,gif,tiff,bmp,svg".to_string(),
            data_dir,
            default_model: env_str("DEFAULT_MODEL", "qwen3.7-plus"),
            upstream_proxy: read_proxy_env(),
            pool_ready_index: true,
            conn_keepalive_seconds: 0,
            token_refresh_interval_hours: 6,
            token_refresh_ahead_days: 7,
            token_refresh_batch_per_cycle: 500,
            token_refresh_jitter_min_ms: 2000,
            token_refresh_jitter_max_ms: 5000,
        }
    }
}

/// Default model aliases. Client-facing model name -> Qwen base model.
pub fn default_model_map() -> HashMap<String, String> {
    let plus = "qwen3.7-plus";
    let flash = "qwen3.5-flash";
    let pairs: &[(&str, &str)] = &[
        // OpenAI
        ("gpt-4o", plus), ("gpt-4o-mini", flash), ("gpt-4-turbo", plus), ("gpt-4", plus),
        ("gpt-4.1", plus), ("gpt-4.1-mini", flash), ("gpt-3.5-turbo", flash),
        ("gpt-5", plus), ("o1", plus), ("o1-mini", flash), ("o3", plus), ("o3-mini", flash),
        // Anthropic
        ("claude-opus-4-6", plus), ("claude-opus-4-8", plus), ("claude-sonnet-4-5", plus),
        ("claude-sonnet-4-6", plus), ("claude-3-opus", plus), ("claude-3.5-sonnet", plus),
        ("claude-3-sonnet", plus), ("claude-3-haiku", flash),
        // Gemini
        ("gemini-2.5-pro", plus), ("gemini-2.5-flash", flash),
        // Qwen aliases
        ("qwen", plus), ("qwen-max", plus), ("qwen-plus", plus), ("qwen-turbo", flash),
        ("qwen3.6-plus", plus),
        // DeepSeek
        ("deepseek-chat", plus), ("deepseek-reasoner", plus),
    ];
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// Resolve an alias to the actual upstream base model.
pub fn resolve_model(map: &HashMap<String, String>, name: &str) -> String {
    map.get(name).cloned().unwrap_or_else(|| name.to_string())
}
