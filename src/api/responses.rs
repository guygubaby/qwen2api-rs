//! OpenAI Responses API：POST /v1/responses、/responses。

use crate::api::openai::sse_response;
use crate::auth::{add_used_tokens, resolve_auth};
use crate::error::AppError;
use crate::execution::{self, OutEvent};
use crate::request::{build_openai_request, StandardRequest};
use crate::state::AppState;
use crate::util::{now_unix, short_id};
use axum::body::{Body, Bytes};
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;

/// 把 Responses 的 input/instructions 轉成 chat messages（再交給通用 builder）。
fn responses_to_chat_body(body: &Value) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    match body.get("input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            for it in items {
                let role = it.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = it.get("content").cloned().unwrap_or(Value::Null);
                messages.push(json!({ "role": role, "content": content }));
            }
        }
        _ => {}
    }
    let system = body.get("instructions").cloned().unwrap_or(Value::Null);
    json!({
        "model": body.get("model").cloned().unwrap_or(json!("gpt-5")),
        "system": system,
        "messages": messages,
        "tools": body.get("tools").cloned().unwrap_or(json!([])),
        "stream": body.get("stream").cloned().unwrap_or(json!(false)),
        "enable_thinking": body.get("reasoning").is_some(),
    })
}

pub async fn create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return AppError::BadRequest(format!("JSON parse error: {e}")).into_response(),
    };
    let auth = match resolve_auth(&state, &headers, &query).await {
        Ok(a) => a,
        Err(e) => return e.into_response(),
    };
    let chat_body = responses_to_chat_body(&body);
    let mut std = build_openai_request(&state, &chat_body, &headers, "responses", "gpt-5").await;
    let token = auth.token;
    std.caller = Some(token.clone());

    if std.stream {
        responses_sse(state, std, token)
    } else {
        let model = std.response_model.clone();
        let registry = execution::registry_for(&std);
        let r = execution::collect_completion(state.clone(), std, registry).await;
        if let Some(err) = &r.error {
            return AppError::Upstream(err.clone()).into_response();
        }
        if r.usage.total_tokens > 0 {
            add_used_tokens(&state, &token, r.usage.total_tokens).await;
        }
        let resp_id = format!("resp_{}", short_id(24));
        Json(build_response_object(&resp_id, &model, &r.content, &r.usage)).into_response()
    }
}

fn build_response_object(id: &str, model: &str, text: &str, usage: &execution::Usage) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now_unix(),
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message",
            "id": format!("msg_{}", short_id(16)),
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
        }],
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    })
}

fn responses_sse(state: AppState, std: StandardRequest, token: String) -> Response {
    let model = std.response_model.clone();
    let registry = execution::registry_for(&std);
    let resp_id = format!("resp_{}", short_id(24));
    let body_stream = async_stream::stream! {
        // response.created
        let created = json!({"type":"response.created","response":{"id": resp_id,"object":"response","model": model,"status":"in_progress"}});
        yield Ok::<_, Infallible>(Bytes::from(format!("event: response.created\ndata: {created}\n\n")));

        let mut s = Box::pin(execution::run_completion(state.clone(), std, registry));
        let mut full_text = String::new();
        let mut billed = 0i64;
        while let Some(ev) = s.next().await {
            match ev {
                OutEvent::ReasoningDelta(r) => {
                    let v = json!({"type":"response.reasoning_text.delta","delta": r});
                    yield Ok(Bytes::from(format!("event: response.reasoning_text.delta\ndata: {v}\n\n")));
                }
                OutEvent::ContentDelta(c) => {
                    full_text.push_str(&c);
                    let v = json!({"type":"response.output_text.delta","delta": c});
                    yield Ok(Bytes::from(format!("event: response.output_text.delta\ndata: {v}\n\n")));
                }
                OutEvent::Done { usage, .. } => {
                    billed = usage.total_tokens;
                    let obj = build_response_object(&resp_id, &model, &full_text, &usage);
                    let v = json!({"type":"response.completed","response": obj});
                    yield Ok(Bytes::from(format!("event: response.completed\ndata: {v}\n\n")));
                    yield Ok(Bytes::from("data: [DONE]\n\n".to_string()));
                }
                OutEvent::Error(e) => {
                    let v = json!({"type":"error","error":{"message": e}});
                    yield Ok(Bytes::from(format!("event: error\ndata: {v}\n\n")));
                }
                _ => {}
            }
        }
        if billed > 0 { add_used_tokens(&state, &token, billed).await; }
    };
    sse_response(Body::from_stream(body_stream))
}
