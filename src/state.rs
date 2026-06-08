//! Shared application state.

use crate::account::AccountPool;
use crate::config::Settings;
use crate::db::JsonDb;
use crate::upstream::{ChatIdPool, Executor, QwenClient};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

pub type AppState = Arc<AppStateInner>;

/// Persistent runtime settings. Currently only stores the upstream proxy.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub upstream_proxy: Option<String>,
}

pub struct AppStateInner {
    pub settings: Settings,
    /// 模型別名映射，可在管理台運行時更新。
    pub model_map: RwLock<HashMap<String, String>>,
    pub pool: Arc<AccountPool>,
    pub client: Arc<QwenClient>,
    pub chat_id_pool: Arc<ChatIdPool>,
    pub executor: Arc<Executor>,
    pub file_store: Arc<crate::context::file_store::FileStore>,
    /// 請求統計子系統（背景批次寫入 SQLite）。
    pub stats: Arc<crate::stats::Stats>,
    /// 媒體任務佇列（圖片/影片背景生成 + 本地保存）。
    pub media_queue: Arc<crate::media::MediaQueue>,
    /// t2v 已知無權限的帳號（持久化跳過集）。
    pub no_t2v: JsonDb<HashSet<String>>,
    /// 快取上游模型列表（避免每次 /v1/models 都打上游）。
    pub upstream_models: RwLock<UpstreamModelsCache>,
    /// 持久化執行期設定（出口代理）。
    runtime_cfg: JsonDb<RuntimeConfig>,
}

#[derive(Default)]
pub struct UpstreamModelsCache {
    pub data: Vec<serde_json::Value>,
    pub fetched_at: f64,
}

impl AppStateInner {
    pub async fn new(settings: Settings) -> AppState {
        let model_map = crate::config::default_model_map();

        if let Some(raw_accounts) = settings.accounts_json.as_deref() {
            match serde_json::from_str::<serde_json::Value>(raw_accounts) {
                Ok(v) if v.is_array() => {
                    crate::db::write_json_atomic(&settings.accounts_file, &v).await;
                    tracing::info!(
                        "Loaded Qwen accounts from QWEN_ACCOUNTS_JSON into {:?}",
                        settings.accounts_file
                    );
                }
                Ok(_) => {
                    tracing::error!("QWEN_ACCOUNTS_JSON must be a JSON array of account objects");
                }
                Err(e) => {
                    tracing::error!("Failed to parse QWEN_ACCOUNTS_JSON: {e}");
                }
            }
        }

        let pool = AccountPool::load(&settings).await;

        // Persistent proxy setting wins over environment configuration.
        let runtime_cfg = JsonDb::load(&settings.config_file, RuntimeConfig::default()).await;
        let initial_proxy = {
            let cfg = runtime_cfg.get().await;
            cfg.upstream_proxy.clone().or_else(|| settings.upstream_proxy.clone())
        };
        let client = Arc::new(QwenClient::new(initial_proxy));
        let chat_id_pool = ChatIdPool::new(
            client.clone(),
            pool.clone(),
            settings.chat_id_prewarm_target_per_account,
            settings.chat_id_prewarm_ttl_seconds,
            settings.chat_id_prewarm_max_accounts,
            settings.default_model.clone(),
        );
        let executor = Arc::new(Executor::new(pool.clone(), client.clone(), chat_id_pool.clone(), &settings));

        let file_store = crate::context::file_store::FileStore::new(
            settings.context_generated_dir.clone(),
            settings.uploaded_files_file.clone(),
        )
        .await;

        let stats = crate::stats::Stats::new(&settings.stats_file);

        let media_store = crate::media::MediaStore::new(&settings.media_db_file, &settings.media_dir);
        let media_queue = crate::media::MediaQueue::new(media_store, settings.media_concurrency, settings.media_max_attempts);

        let no_t2v = JsonDb::load(&settings.no_t2v_file, HashSet::<String>::new()).await;

        Arc::new(AppStateInner {
            model_map: RwLock::new(model_map),
            pool,
            client,
            chat_id_pool,
            executor,
            file_store,
            stats,
            media_queue,
            no_t2v,
            upstream_models: RwLock::new(UpstreamModelsCache::default()),
            runtime_cfg,
            settings,
        })
    }

    /// Set the upstream proxy immediately and persist it.
    pub async fn set_upstream_proxy(&self, proxy: Option<String>) {
        let normalized = proxy.and_then(|p| {
            let t = p.trim().to_string();
            if t.is_empty() { None } else { Some(t) }
        });
        self.client.set_proxy(normalized.clone());
        self.runtime_cfg.set(RuntimeConfig { upstream_proxy: normalized }).await;
    }

    /// Resolve a model alias. Unknown non-Qwen names fall back to the default
    /// model so downstream aliases are not sent to Qwen unchanged.
    pub async fn resolve_model(&self, name: &str) -> String {
        let resolved = {
            let map = self.model_map.read().await;
            crate::config::resolve_model(&map, name)
        };
        if resolved.to_lowercase().starts_with("qwen") {
            resolved
        } else {
            self.settings.default_model.clone()
        }
    }
}
