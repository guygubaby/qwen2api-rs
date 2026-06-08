//! Embeddings（佔位）：POST /v1/embeddings、/embeddings。
//! 對齊原版：由 sha256(text) 生成確定性 1536 維向量（非真實語意向量）。

use crate::auth::resolve_auth;
use crate::error::AppError;
use crate::state::AppState;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

const DIM: usize = 1536;

fn fake_embedding(text: &str) -> Vec<f32> {
    let mut out = Vec::with_capacity(DIM);
    let mut counter: u32 = 0;
    while out.len() < DIM {
        let mut h = Sha256::new();
        h.update(text.as_bytes());
        h.update(counter.to_le_bytes());
        let digest = h.finalize();
        for chunk in digest.chunks(4) {
            if out.len() >= DIM {
                break;
            }
            let mut b = [0u8; 4];
            b[..chunk.len()].copy_from_slice(chunk);
            let v = u32::from_le_bytes(b) as f64 / u32::MAX as f64;
            out.push(((v * 2.0) - 1.0) as f32); // 映射到 [-1,1]
        }
        counter += 1;
    }
    out
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
    if let Err(e) = resolve_auth(&state, &headers, &query).await {
        return e.into_response();
    }
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
        _ => vec![],
    };
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("text-embedding-3-small");
    let mut total_chars = 0usize;
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, text)| {
            total_chars += text.chars().count();
            json!({ "object": "embedding", "index": i, "embedding": fake_embedding(text) })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": { "prompt_tokens": total_chars / 4, "total_tokens": total_chars / 4 },
    }))
    .into_response()
}
