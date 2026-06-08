//! Qwen 上游帳號結構，對應 Python `core/account_pool/pool_core.py` 的 Account。
//! 持久欄位會寫進 accounts.json；執行期欄位以 serde(skip) + default 處理。

use crate::util::now_secs;
use serde::{Deserialize, Serialize};

fn default_valid() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub email: String,
    #[serde(default)]
    pub password: String,
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub cookies: String,
    #[serde(default)]
    pub username: String,

    #[serde(default)]
    pub activation_pending: bool,
    #[serde(default = "default_status")]
    pub status_code: String,
    #[serde(default)]
    pub last_error: String,

    #[serde(default)]
    pub last_request_started: f64,
    #[serde(default)]
    pub last_request_finished: f64,
    #[serde(default)]
    pub consecutive_failures: i64,
    #[serde(default)]
    pub rate_limit_strikes: i64,

    // ---- 執行期欄位（不持久化）----
    #[serde(skip, default = "default_valid")]
    pub valid: bool,
    #[serde(skip)]
    pub inflight: i64,
    #[serde(skip)]
    pub rate_limited_until: f64,
    #[serde(skip)]
    pub last_used: f64,
    #[serde(skip)]
    pub healing: bool,
}

fn default_status() -> String {
    "valid".to_string()
}

impl Account {
    pub fn new(email: String, password: String, token: String, cookies: String, username: String) -> Self {
        let activation_pending = false;
        Account {
            email,
            password,
            token,
            cookies,
            username,
            activation_pending,
            status_code: if activation_pending { "pending_activation" } else { "valid" }.to_string(),
            last_error: String::new(),
            last_request_started: 0.0,
            last_request_finished: 0.0,
            consecutive_failures: 0,
            rate_limit_strikes: 0,
            valid: !activation_pending,
            inflight: 0,
            rate_limited_until: 0.0,
            last_used: 0.0,
            healing: false,
        }
    }

    /// 載入後初始化執行期欄位（serde skip 的欄位不會自動設對）。
    pub fn init_runtime(&mut self) {
        self.valid = !self.activation_pending && self.status_code != "banned" && self.status_code != "auth_error";
        // 由持久 status_code 推回 valid
        if self.status_code == "valid" {
            self.valid = true;
        }
        self.inflight = 0;
        self.rate_limited_until = 0.0;
        self.last_used = 0.0;
        self.healing = false;
    }

    pub fn is_rate_limited(&self) -> bool {
        self.rate_limited_until > now_secs()
    }

    pub fn is_available(&self, min_interval_ms: u64) -> bool {
        self.valid && !self.is_rate_limited() && self.next_available_at(min_interval_ms) <= now_secs()
    }

    /// 下一次可用時間（風控強制休息）：限流到期、上次「起始」、上次「結束」各 + 最小間隔，取最大。
    /// 從結束起算可確保長請求（如出圖 30s）完成後仍有完整休息。
    pub fn next_available_at(&self, min_interval_ms: u64) -> f64 {
        let interval = min_interval_ms as f64 / 1000.0;
        let after_started = self.last_request_started + interval;
        let after_finished = self.last_request_finished + interval;
        f64::max(self.rate_limited_until, f64::max(after_started, after_finished))
    }

    /// 動態狀態碼（對齊 Python get_status_code）。
    pub fn get_status_code(&self) -> String {
        if self.status_code == "banned" {
            return "banned".to_string();
        }
        if self.is_rate_limited() {
            return "rate_limited".to_string();
        }
        if self.activation_pending {
            return "pending_activation".to_string();
        }
        if !self.valid {
            if self.status_code == "valid" || self.status_code.is_empty() {
                return "auth_error".to_string();
            }
            return self.status_code.clone();
        }
        "valid".to_string()
    }

    pub fn get_status_text(&self) -> String {
        match self.get_status_code().as_str() {
            "valid" => "Healthy",
            "pending_activation" => "Pending activation",
            "rate_limited" => "Rate limited",
            "banned" => "Banned",
            "auth_error" => "Authentication failed",
            "invalid" => "Invalid",
            _ => "Unknown",
        }
        .to_string()
    }

    /// 管理台列表用的序列化（含執行期狀態）。
    pub fn to_admin_json(&self) -> serde_json::Value {
        serde_json::json!({
            "email": self.email,
            "username": self.username,
            "valid": self.valid,
            "inflight": self.inflight,
            "rate_limited_until": self.rate_limited_until,
            "activation_pending": self.activation_pending,
            "status_code": self.get_status_code(),
            "status_text": self.get_status_text(),
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "rate_limit_strikes": self.rate_limit_strikes,
        })
    }
}
