//! 影像生成：POST /v1/images/generations、/images/generations。
//! 驅動 Qwen t2i chat_type。走 media::generate_with_retry（應用層重試 + 帳號輪換）；
//! 成功後在背景下載本地備份（API 仍回 CDN URL）。

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
    let n = body.get("n").and_then(|v| v.as_u64()).unwrap_or(1).clamp(1, 4);
    let ratio = body
        .get("ratio")
        .or_else(|| body.get("aspect_ratio"))
        .and_then(|v| v.as_str())
        .unwrap_or("1:1")
        .to_string();
    let size = body.get("size").and_then(|v| v.as_str()).unwrap_or("1024x1024").to_string();
    let width = body.get("width").and_then(|v| v.as_i64());
    let height = body.get("height").and_then(|v| v.as_i64());

    let options = ImageOptions {
        size: Some(size.clone()),
        ratio: Some(ratio.clone()),
        width,
        height,
    };

    let mut data_items: Vec<Value> = Vec::new();
    let mut all_urls: Vec<String> = Vec::new();
    let mut last_err: Option<String> = None;

    for _ in 0..n {
        let out = media::generate_with_retry(
            &state,
            &prompt,
            MediaKind::Image,
            options.clone(),
            state.settings.media_max_attempts,
            caller.clone(),
        )
        .await;
        if out.urls.is_empty() {
            last_err = out.error;
            continue;
        }
        for u in out.urls {
            all_urls.push(u.clone());
            data_items.push(json!({
                "url": u,
                "revised_prompt": prompt,
                "ratio": ratio,
                "size": size,
                "width": width,
                "height": height,
            }));
        }
    }

    if data_items.is_empty() {
        let msg = last_err.unwrap_or_else(|| "No image was generated; upstream did not return an image URL".into());
        return AppError::Upstream(msg).into_response();
    }

    // 背景本地備份 + 記錄媒體庫（API 仍回 CDN URL）
    spawn_backup(state.clone(), all_urls, prompt.clone(), ratio.clone(), size, caller);

    Json(json!({ "created": now_unix(), "data": data_items })).into_response()
}

fn spawn_backup(state: AppState, urls: Vec<String>, prompt: String, ratio: String, size: String, caller: Option<String>) {
    tokio::spawn(async move {
        let client = state.client.client();
        let results = state.media_queue.store.backup_urls(&client, &urls, MediaKind::Image).await;
        let params = json!({ "ratio": ratio, "size": size });
        state
            .media_queue
            .store
            .insert_done(MediaKind::Image, &prompt, params, results, caller)
            .await;
    });
}
