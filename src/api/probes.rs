//! Health/readiness probes.

use crate::state::AppState;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

pub async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

pub async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    let accounts = state.pool.count().await;
    Json(json!({ "status": "ready", "accounts": accounts }))
}

pub async fn root() -> impl IntoResponse {
    Json(json!({
        "status": "qwen2API Enterprise Gateway (Rust) is running",
        "version": "2.0.0"
    }))
}
