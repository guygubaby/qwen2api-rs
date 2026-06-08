//! 統一錯誤型別，可轉成 axum 回應（OpenAI 風格 error JSON）。

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("payment required: {0}")]
    QuotaExceeded(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("upstream error: {0}")]
    Upstream(String),
    #[error("{0}")]
    Internal(String),
}

impl AppError {
    pub fn status(&self) -> StatusCode {
        match self {
            AppError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            AppError::Forbidden(_) => StatusCode::FORBIDDEN,
            AppError::QuotaExceeded(_) => StatusCode::PAYMENT_REQUIRED,
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            AppError::Upstream(_) => StatusCode::BAD_GATEWAY,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    pub fn message(&self) -> String {
        match self {
            AppError::Unauthorized(m)
            | AppError::Forbidden(m)
            | AppError::QuotaExceeded(m)
            | AppError::BadRequest(m)
            | AppError::NotFound(m)
            | AppError::NotImplemented(m)
            | AppError::Upstream(m)
            | AppError::Internal(m) => m.clone(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        let msg = self.message();
        let body = Json(json!({
            "error": {
                "message": msg,
                "type": "qwen2api_error",
                "code": status.as_u16(),
            },
            // 兼容部分前端讀 detail / 頂層 error 字串
            "detail": msg,
        }));
        (status, body).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::BadRequest(format!("JSON parse error: {e}"))
    }
}

pub type AppResult<T> = Result<T, AppError>;
