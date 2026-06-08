//! 影片生成：POST /v1/videos/generations、/videos/generations。
//! 驅動 Qwen t2v chat_type。影片較慢且帳號有「每日額度上限」，故走 media::generate_with_retry
//! 做應用層重試 + 帳號輪換；成功後額外在背景下載一份本地備份（API 仍回 CDN URL）。

use crate::auth::resolve_auth;
use crate::error::AppError;
use crate::media::{self, MediaKind};
use crate::state::AppState;
use crate::upstream::ImageOptions;
use crate::util::now_unix;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::HashMap;

pub async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return AppError::BadRequest(format!("JSON parse error: {e}")).into_response(),
    };
    let caller = match resolve_auth(&state, &headers, &query).await {
        Ok(a) => Some(a.token),
        Err(e) => return e.into_response(),
    };
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return AppError::BadRequest("prompt is required".into()).into_response();
    }
    let ratio = body
        .get("ratio")
        .or_else(|| body.get("aspect_ratio"))
        .and_then(|v| v.as_str())
        .unwrap_or("16:9")
        .to_string();

    let options = ImageOptions { size: None, ratio: Some(ratio.clone()), width: None, height: None };
    let out = media::generate_with_retry(
        &state,
        &prompt,
        MediaKind::Video,
        options,
        state.settings.media_max_attempts,
        caller.clone(),
    )
    .await;

    if out.urls.is_empty() {
        return AppError::Upstream(out.error.unwrap_or_else(|| {
            "Video generation failed after rotating accounts; all accounts may be out of quota".into()
        }))
        .into_response();
    }

    // 背景：下載本地備份 + 記錄到媒體庫（API 回應不等待，仍回 CDN URL）
    spawn_backup(state.clone(), out.urls.clone(), prompt.clone(), ratio.clone(), caller);

    let data: Vec<Value> = out.urls.into_iter().map(|u| json!({ "url": u, "revised_prompt": "", "ratio": ratio })).collect();
    Json(json!({ "created": now_unix(), "data": data })).into_response()
}

/// fire-and-forget：下載 CDN 影片到本地並在媒體庫記一筆 done（供畫廊與防丟失）。
fn spawn_backup(state: AppState, urls: Vec<String>, prompt: String, ratio: String, caller: Option<String>) {
    tokio::spawn(async move {
        let client = state.client.client();
        let results = state.media_queue.store.backup_urls(&client, &urls, MediaKind::Video).await;
        let params = json!({ "ratio": ratio });
        state
            .media_queue
            .store
            .insert_done(MediaKind::Video, &prompt, params, results, caller)
            .await;
    });
}
