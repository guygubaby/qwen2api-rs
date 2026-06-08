//! Anthropic Messages：POST /v1/messages、/messages、/anthropic/v1/messages。
//! 以及 count_tokens（開放）。

use crate::api::openai::sse_response;
use crate::auth::{add_used_tokens, resolve_auth};
use crate::error::AppError;
use crate::execution::{self, formatters, presenter::AnthropicStreamTranslator, OutEvent};
use crate::request::build_openai_request;
use crate::state::AppState;
use axum::body::{Body, Bytes};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::convert::Infallible;

pub async fn messages(
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
    let mut std = build_openai_request(&state, &body, &headers, "anthropic", "claude-3-5-sonnet").await;
    let token = auth.token;
    std.caller = Some(token.clone());
    let tool_name_preview: Vec<String> = std.tool_names.iter().take(20).cloned().collect();
    tracing::debug!(
        target: "qwen2api.anthropic",
        model = %std.response_model,
        resolved_model = %std.resolved_model,
        stream = std.stream,
        tools = std.tools.len(),
        profile = %std.client_profile,
        thinking_enabled = ?std.thinking_enabled,
        has_thinking_field = body.get("thinking").is_some(),
        tool_names = ?tool_name_preview,
        "received Anthropic Messages request"
    );

    if std.stream {
        let model = std.response_model.clone();
        let prompt_tokens = execution::estimate_tokens_fast(&std.prompt) as i64;
        let registry = execution::registry_for(&std);
        let body_stream = async_stream::stream! {
            let mut tr = AnthropicStreamTranslator::new(&model, prompt_tokens);
            let mut s = Box::pin(execution::run_completion(state.clone(), std, registry));
            let mut billed = 0i64;
            while let Some(ev) = s.next().await {
                if let OutEvent::Done { usage, .. } = &ev { billed = usage.total_tokens; }
                for line in tr.on_event(&ev) {
                    yield Ok::<_, Infallible>(Bytes::from(line));
                }
            }
            if billed > 0 { add_used_tokens(&state, &token, billed).await; }
        };
        sse_response(Body::from_stream(body_stream))
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
        Json(formatters::build_anthropic_message(&model, &result)).into_response()
    }
}

/// POST .../messages/count_tokens（開放，不需認證）。回 input_tokens（×1.35 提前觸發壓縮）。
pub async fn count_tokens(body: Bytes) -> Response {
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap_or(serde_json::json!({}));
    // 粗估：把 system + 所有 messages 文字串起來算 token
    let mut text = String::new();
    if let Some(sys) = body.get("system") {
        text.push_str(&crate::request::prompt_builder::extract_text_content(sys));
    }
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for m in msgs {
            if let Some(c) = m.get("content") {
                text.push_str(&crate::request::prompt_builder::extract_text_content(c));
                text.push('\n');
            }
        }
    }
    let base = execution::count_tokens(&text) as f64;
    let inflated = (base * 1.35) as i64;
    Json(serde_json::json!({ "input_tokens": inflated })).into_response()
}
