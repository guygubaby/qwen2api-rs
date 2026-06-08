//! Gemini 相容：generateContent（非串流）、streamGenerateContent（串流）。
//! 路徑形如 /v1beta/models/{model}:generateContent。

use crate::api::openai::sse_response;
use crate::auth::{add_used_tokens, resolve_auth};
use crate::error::AppError;
use crate::execution::{self, formatters, presenter, OutEvent};
use crate::request::build_gemini_request;
use crate::state::AppState;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::convert::Infallible;

/// 從 "{model}:{action}" 拆出 (model, action)。
fn split_model_action(raw: &str) -> (String, String) {
    match raw.rsplit_once(':') {
        Some((m, a)) => (m.to_string(), a.to_string()),
        None => (raw.to_string(), "generateContent".to_string()),
    }
}

pub async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model_action): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let (model, action) = split_model_action(&model_action);
    let body: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return AppError::BadRequest(format!("JSON parse error: {e}")).into_response(),
    };
    if let Err(e) = resolve_auth(&state, &headers, &query).await {
        return e.into_response();
    }
    let token_holder = crate::auth::extract_api_token(&headers, &query).unwrap_or_default();
    let mut std = build_gemini_request(&state, &body, &headers, &model).await;
    std.caller = Some(token_holder.clone());
    let is_stream = action.contains("stream");

    if is_stream {
        let registry = execution::registry_for(&std);
        let body_stream = async_stream::stream! {
            let mut s = Box::pin(execution::run_completion(state.clone(), std, registry));
            let mut billed = 0i64;
            while let Some(ev) = s.next().await {
                match ev {
                    OutEvent::ContentDelta(c) => {
                        yield Ok::<_, Infallible>(Bytes::from(presenter::gemini_text_chunk(&c)));
                    }
                    OutEvent::Done { usage, .. } => {
                        billed = usage.total_tokens;
                        yield Ok(Bytes::from(presenter::gemini_final_chunk(&usage)));
                    }
                    OutEvent::Error(e) => {
                        yield Ok(Bytes::from(presenter::gemini_error_chunk(&e)));
                    }
                    _ => {}
                }
            }
            if billed > 0 { add_used_tokens(&state, &token_holder, billed).await; }
        };
        sse_response(Body::from_stream(body_stream))
    } else {
        let registry = execution::registry_for(&std);
        let result = execution::collect_completion(state.clone(), std, registry).await;
        if let Some(err) = &result.error {
            return AppError::Upstream(err.clone()).into_response();
        }
        if result.usage.total_tokens > 0 {
            add_used_tokens(&state, &token_holder, result.usage.total_tokens).await;
        }
        Json(formatters::build_gemini_generate(&result)).into_response()
    }
}
