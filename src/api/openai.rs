//! OpenAI Chat Completions：POST /v1/chat/completions、/chat/completions。

use crate::auth::{add_used_tokens, resolve_auth};
use crate::error::AppError;
use crate::execution::{self, formatters, translator::OpenAiStreamTranslator, OutEvent};
use crate::request::{build_openai_request, StandardRequest};
use crate::state::AppState;
use axum::body::{Body, Bytes};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::convert::Infallible;

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let body: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return AppError::BadRequest(format!("JSON parse error: {e}")).into_response(),
    };
    let auth = match resolve_auth(&state, &headers, &query).await {
        Ok(a) => a,
        Err(e) => return e.into_response(),
    };
    let mut std = build_openai_request(&state, &body, &headers, "openai", "gpt-3.5-turbo").await;
    std.caller = Some(auth.token.clone());
    run_openai(state, std, auth.token).await
}

/// 共用：依 stream 決定串流或 JSON 回應。
pub async fn run_openai(state: AppState, std: StandardRequest, token: String) -> Response {
    if std.stream {
        openai_sse_response(state, std, token)
    } else {
        let model = std.response_model.clone();
        let registry = execution::registry_for(&std);
        let result = execution::collect_completion(state.clone(), std, registry).await;
        if let Some(err) = &result.error {
            return AppError::Upstream(err.clone()).into_response();
        }
        if result.usage.total_tokens > 0 {
            add_used_tokens(&state, &token, result.usage.total_tokens).await;
        }
        Json(formatters::build_openai_completion(&model, &result)).into_response()
    }
}

fn openai_sse_response(state: AppState, std: StandardRequest, token: String) -> Response {
    let model = std.response_model.clone();
    let registry = execution::registry_for(&std);
    let body_stream = async_stream::stream! {
        let mut translator = OpenAiStreamTranslator::new(&model);
        let mut s = Box::pin(execution::run_completion(state.clone(), std, registry));
        let mut billed = 0i64;
        while let Some(ev) = s.next().await {
            if let OutEvent::Done { usage, .. } = &ev {
                billed = usage.total_tokens;
            }
            for line in translator.on_event(&ev) {
                yield Ok::<_, Infallible>(Bytes::from(line));
            }
        }
        if billed > 0 {
            add_used_tokens(&state, &token, billed).await;
        }
    };
    sse_response(Body::from_stream(body_stream))
}

/// 建立 text/event-stream 回應。
pub fn sse_response(body: Body) -> Response {
    Response::builder()
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-accel-buffering", "no")
        .body(body)
        .unwrap()
}
