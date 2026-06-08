//! 檔案上傳：POST /v1/files、/api/files/upload（multipart）；DELETE /v1/files/{id}。
//! 存本地，回傳 file_id 與可重用的 content_block；實際 OSS 上傳在 chat 時進行。

use crate::auth::resolve_auth;
use crate::error::AppError;
use crate::state::AppState;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::collections::HashMap;

pub async fn upload(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    mut multipart: Multipart,
) -> Response {
    if let Err(e) = resolve_auth(&state, &headers, &query).await {
        return e.into_response();
    }
    let mut filename = "file".to_string();
    let mut content_type = "application/octet-stream".to_string();
    let mut bytes: Vec<u8> = Vec::new();
    let mut purpose = "user".to_string();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "purpose" {
            purpose = field.text().await.unwrap_or_default();
            continue;
        }
        if name == "file" || name.is_empty() {
            if let Some(fname) = field.file_name() {
                filename = fname.to_string();
            }
            if let Some(ct) = field.content_type() {
                content_type = ct.to_string();
            }
            match field.bytes().await {
                Ok(b) => bytes = b.to_vec(),
                Err(e) => return AppError::BadRequest(format!("Failed to read file: {e}")).into_response(),
            }
        }
    }

    if bytes.is_empty() {
        return AppError::BadRequest("No file content provided".into()).into_response();
    }

    // 副檔名白名單檢查
    let allowed = &state.settings.context_allowed_user_exts;
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
    if !ext.is_empty() && !allowed.split(',').any(|e| e.trim() == ext) {
        return AppError::BadRequest(format!("Unsupported file extension: .{ext}")).into_response();
    }

    let meta = state.file_store.save_bytes(&filename, &content_type, &bytes, &purpose).await;
    Json(json!({
        "id": meta.id,
        "object": "file",
        "bytes": meta.size,
        "created_at": meta.created_at,
        "filename": meta.filename,
        "purpose": meta.purpose,
        "content_type": meta.content_type,
        // 方便客戶端直接放進 messages 的附件區塊
        "content_block": {
            "type": "input_file",
            "file_id": meta.id,
            "filename": meta.filename,
            "mime_type": meta.content_type,
        },
    }))
    .into_response()
}

pub async fn delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Path(file_id): Path<String>,
) -> Response {
    if let Err(e) = resolve_auth(&state, &headers, &query).await {
        return e.into_response();
    }
    let ok = state.file_store.delete(&file_id).await;
    Json(json!({ "id": file_id, "object": "file", "deleted": ok })).into_response()
}
