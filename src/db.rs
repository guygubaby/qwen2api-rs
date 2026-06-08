//! 帶鎖的 JSON 檔案存儲，對應 Python `core/database.py` 的 AsyncJsonDB。
//! 泛型化 JSON store. Writes use a temporary file + rename for atomic replacement.

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

/// 原子寫入任意可序列化值到 JSON 檔（臨時檔 + rename）。
pub async fn write_json_atomic<T: Serialize>(path: &Path, value: &T) {
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    match serde_json::to_vec_pretty(value) {
        Ok(bytes) => {
            let tmp = path.with_extension("json.tmp");
            if tokio::fs::write(&tmp, &bytes).await.is_ok() {
                let _ = tokio::fs::rename(&tmp, path).await;
            }
        }
        Err(e) => tracing::error!("write_json_atomic serialization failed: {e}"),
    }
}

/// 讀取 JSON 檔；失敗回 default。
pub async fn read_json_or<T: DeserializeOwned>(path: &Path, default: T) -> T {
    match tokio::fs::read(path).await {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or(default),
        Err(_) => default,
    }
}

#[derive(Clone)]
pub struct JsonDb<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + 'static,
{
    path: PathBuf,
    data: Arc<Mutex<T>>,
}

impl<T> JsonDb<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + 'static,
{
    /// 載入檔案；不存在或parse failed則用 default，並寫回一份。
    pub async fn load(path: impl AsRef<Path>, default: T) -> Self {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let data = match tokio::fs::read(&path).await {
            Ok(bytes) => match serde_json::from_slice::<T>(&bytes) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("JsonDb parse failed {:?}: {e}, using default value", path);
                    default
                }
            },
            Err(_) => default,
        };
        let db = JsonDb { path, data: Arc::new(Mutex::new(data)) };
        db.save().await; // 確保檔案存在
        db
    }

    /// 取得資料副本。
    pub async fn get(&self) -> T {
        self.data.lock().await.clone()
    }

    /// 以新值整體覆蓋並持久化。
    pub async fn set(&self, value: T) {
        {
            let mut guard = self.data.lock().await;
            *guard = value;
        }
        self.save().await;
    }

    /// 在鎖內就地修改並持久化（避免讀-改-寫競態）。
    pub async fn update<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let r;
        let snapshot;
        {
            let mut guard = self.data.lock().await;
            r = f(&mut guard);
            snapshot = guard.clone();
        }
        self.write_to_disk(&snapshot).await;
        r
    }

    /// 將目前記憶體值寫回磁碟。
    pub async fn save(&self) {
        let snapshot = self.data.lock().await.clone();
        self.write_to_disk(&snapshot).await;
    }

    async fn write_to_disk(&self, value: &T) {
        // ensure_ascii=false + indent=2：serde_json to_string_pretty 即為 UTF-8 不轉義。
        match serde_json::to_vec_pretty(value) {
            Ok(bytes) => {
                let tmp = self.path.with_extension("json.tmp");
                if let Err(e) = tokio::fs::write(&tmp, &bytes).await {
                    tracing::error!("JsonDb temporary file write failed {:?}: {e}", tmp);
                    return;
                }
                if let Err(e) = tokio::fs::rename(&tmp, &self.path).await {
                    tracing::error!("JsonDb rename failed {:?}: {e}", self.path);
                }
            }
            Err(e) => tracing::error!("JsonDb serialization failed: {e}"),
        }
    }
}
