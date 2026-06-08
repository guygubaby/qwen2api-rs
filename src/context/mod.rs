//! 附件 / 上下文處理，對應 Python `services/attachment_preprocessor.py` + `context_attachment_manager.py`（精簡）。
//! 策略：小文字檔內聯進 prompt（免 OSS、最快）；其餘（二進位/圖片/大檔）走 OSS 上傳並綁定帳號。

pub mod file_store;
pub mod oss;

use crate::state::AppState;
use base64::Engine;
use serde_json::Value;

/// 附件預處理結果。
#[derive(Debug, Default)]
pub struct AttachmentResult {
    /// 上傳到上游的 remote_ref，放進 payload.messages[].files。
    pub files: Vec<Value>,
    /// 內聯到 prompt 的文字（小文字檔）。
    pub inline_note: String,
    /// 綁定帳號（上傳檔案的帳號，後續對話須用同一帳號）。
    pub bound_account: Option<String>,
}

struct RawAttachment {
    filename: String,
    content_type: String,
    bytes: Vec<u8>,
}

fn is_textual(content_type: &str, filename: &str) -> bool {
    if content_type.starts_with("text/") {
        return true;
    }
    if matches!(content_type, "application/json" | "application/xml" | "application/x-yaml" | "application/javascript") {
        return true;
    }
    let lower = filename.to_lowercase();
    [
        ".txt", ".md", ".json", ".log", ".xml", ".yaml", ".yml", ".csv", ".html", ".css", ".py", ".js", ".ts",
        ".java", ".c", ".cpp", ".cs", ".php", ".go", ".rb", ".sh", ".rs", ".toml", ".ini",
    ]
    .iter()
    .any(|e| lower.ends_with(e))
}

/// 解析 data: URI → (content_type, bytes)。
fn parse_data_uri(url: &str) -> Option<(String, Vec<u8>)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if meta.contains(";base64") {
        let ct = meta.split(';').next().unwrap_or("application/octet-stream").to_string();
        let bytes = base64::engine::general_purpose::STANDARD.decode(data).ok()?;
        Some((ct, bytes))
    } else {
        let ct = meta.split(';').next().unwrap_or("text/plain").to_string();
        Some((ct, urlencoding::decode(data).ok()?.into_owned().into_bytes()))
    }
}

/// 從一個 content part 抽出附件（若有）。
async fn extract_from_part(state: &AppState, part: &Value) -> Option<RawAttachment> {
    let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ptype {
        "input_file" | "file" => {
            // 依 file_id 從本地檔案庫取回
            if let Some(fid) = part.get("file_id").and_then(|v| v.as_str()) {
                if let Some((meta, bytes)) = state.file_store.get(fid).await {
                    return Some(RawAttachment { filename: meta.filename, content_type: meta.content_type, bytes });
                }
            }
            // 內嵌 data（部分客戶端）
            if let Some(src) = part.get("file_data").or_else(|| part.get("data")).and_then(|v| v.as_str()) {
                if let Some((ct, bytes)) = parse_data_uri(src) {
                    let name = part.get("filename").and_then(|v| v.as_str()).unwrap_or("file").to_string();
                    return Some(RawAttachment { filename: name, content_type: ct, bytes });
                }
            }
            None
        }
        "image_url" => {
            let url = part.get("image_url").and_then(|u| u.get("url")).and_then(|v| v.as_str())?;
            if url.starts_with("data:") {
                let (ct, bytes) = parse_data_uri(url)?;
                Some(RawAttachment { filename: format!("image.{}", ct.rsplit('/').next().unwrap_or("png")), content_type: ct, bytes })
            } else {
                None // http URL：留給 prompt 文字保底（prompt_builder 已處理）
            }
        }
        // Anthropic image: {type:image, source:{type:base64, media_type, data}}
        "image" => {
            let src = part.get("source")?;
            if src.get("type").and_then(|v| v.as_str()) == Some("base64") {
                let mt = src.get("media_type").and_then(|v| v.as_str()).unwrap_or("image/png").to_string();
                let data = src.get("data").and_then(|v| v.as_str())?;
                let bytes = base64::engine::general_purpose::STANDARD.decode(data).ok()?;
                return Some(RawAttachment { filename: format!("image.{}", mt.rsplit('/').next().unwrap_or("png")), content_type: mt, bytes });
            }
            None
        }
        _ => None,
    }
}

/// 掃描 body.messages 收集所有附件。
async fn collect_attachments(state: &AppState, body: &Value) -> Vec<RawAttachment> {
    let mut out = Vec::new();
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for m in msgs {
            if let Some(parts) = m.get("content").and_then(|c| c.as_array()) {
                for p in parts {
                    if let Some(att) = extract_from_part(state, p).await {
                        out.push(att);
                    }
                }
            }
        }
    }
    out
}

/// 主入口：預處理附件。無附件時極快返回。
pub async fn prepare_attachments(state: &AppState, body: &Value) -> AttachmentResult {
    let attachments = collect_attachments(state, body).await;
    if attachments.is_empty() {
        return AttachmentResult::default();
    }

    let mut result = AttachmentResult::default();
    let inline_max = state.settings.context_inline_max_chars;

    // 需要 OSS 的附件才取帳號
    let mut need_account = false;
    for a in &attachments {
        if !(is_textual(&a.content_type, &a.filename) && a.bytes.len() <= inline_max * 3) {
            need_account = true;
        }
    }
    let account = if need_account {
        state.pool.any_valid_account().await
    } else {
        None
    };

    for a in attachments {
        let textual = is_textual(&a.content_type, &a.filename);
        if textual {
            if let Ok(text) = String::from_utf8(a.bytes.clone()) {
                if text.chars().count() <= inline_max {
                    result
                        .inline_note
                        .push_str(&format!("\n\n[Attached file: {}]\n```\n{}\n```", a.filename, text));
                    continue;
                }
            }
        }
        // 走 OSS 上傳
        if let Some((email, token)) = &account {
            match oss::upload_file(
                &state.client.client(),
                token,
                &a.filename,
                &a.bytes,
                &a.content_type,
                state.settings.context_upload_parse_timeout_seconds,
            )
            .await
            {
                Ok(remote_ref) => {
                    result.files.push(remote_ref);
                    result.bound_account = Some(email.clone());
                }
                Err(e) => {
                    tracing::warn!("attachment upload failed {}: {e}", a.filename);
                    result.inline_note.push_str(&format!("\n\n[Attachment {} upload failed and was skipped]", a.filename));
                }
            }
        }
    }

    result
}
