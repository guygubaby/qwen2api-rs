//! 請求構建層：把各協議的 body 轉成內部 StandardRequest。

pub mod client_profiles;
pub mod model_catalog;
pub mod model_modes;
pub mod prompt_builder;
pub mod standard;

pub use standard::StandardRequest;

use crate::state::AppState;
use crate::upstream::ImageOptions;
use axum::http::HeaderMap;
use model_modes::parse_model_mode;
use serde_json::Value;

/// 取最後一則 user 訊息的純文字（影像/影片生成用）。
fn last_user_text(body: &Value) -> Option<String> {
    let msgs = body.get("messages")?.as_array()?;
    for m in msgs.iter().rev() {
        if m.get("role").and_then(|v| v.as_str()) == Some("user") {
            let t = prompt_builder::extract_text_content(m.get("content").unwrap_or(&Value::Null));
            if !t.trim().is_empty() {
                return Some(t);
            }
        }
    }
    None
}

fn coerce_bool(v: Option<&Value>) -> Option<bool> {
    match v {
        Some(Value::Bool(b)) => Some(*b),
        Some(Value::String(s)) => match s.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        },
        Some(Value::Number(n)) => Some(n.as_i64().unwrap_or(0) != 0),
        _ => None,
    }
}

/// 從 body 萃取思考開關（對應 build_chat_standard_request 的 _extract_thinking_enabled）。
fn extract_thinking(body: &Value, force_thinking: bool) -> (Option<bool>, bool) {
    if force_thinking {
        return (Some(true), true);
    }
    // Anthropic thinking 物件 {type:"enabled"|"disabled"}
    if let Some(obj) = body.get("thinking").and_then(|v| v.as_object()) {
        if let Some(t) = obj.get("type").and_then(|v| v.as_str()) {
            return (Some(t == "enabled"), false);
        }
    }
    // 常見布林欄位
    for key in ["enable_thinking", "thinking", "include_reasoning"] {
        if let Some(b) = coerce_bool(body.get(key)) {
            return (Some(b), false);
        }
    }
    // OpenAI reasoning_effort：非 none 視為思考
    if let Some(eff) = body.get("reasoning_effort").and_then(|v| v.as_str()) {
        return (Some(eff != "none"), false);
    }
    (None, false)
}

/// Gemini generateContent → StandardRequest。把 Gemini body 轉成內部 openai-ish body 再壓平。
pub async fn build_gemini_request(
    state: &AppState,
    body: &Value,
    headers: &HeaderMap,
    model_name: &str,
) -> StandardRequest {
    use serde_json::json;
    // 轉訊息
    let mut messages: Vec<Value> = Vec::new();
    if let Some(contents) = body.get("contents").and_then(|v| v.as_array()) {
        for c in contents {
            let role = match c.get("role").and_then(|v| v.as_str()) {
                Some("model") => "assistant",
                _ => "user",
            };
            let mut text = String::new();
            if let Some(parts) = c.get("parts").and_then(|v| v.as_array()) {
                for p in parts {
                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                        text.push_str(t);
                    } else if let Some(fc) = p.get("functionCall") {
                        text.push_str(&format!("[tool call {}]", fc));
                    } else if let Some(fr) = p.get("functionResponse") {
                        text.push_str(&format!("[tool result {}]", fr));
                    }
                }
            }
            messages.push(json!({ "role": role, "content": text }));
        }
    }
    // systemInstruction
    let system = body
        .get("systemInstruction")
        .or_else(|| body.get("system_instruction"))
        .and_then(|si| si.get("parts"))
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|x| x.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    // tools: functionDeclarations → 扁平工具列表
    let mut tools: Vec<Value> = Vec::new();
    if let Some(tarr) = body.get("tools").and_then(|v| v.as_array()) {
        for t in tarr {
            if let Some(fds) = t.get("functionDeclarations").and_then(|v| v.as_array()) {
                for fd in fds {
                    tools.push(fd.clone());
                }
            }
        }
    }

    let synthetic = json!({ "system": system, "messages": messages, "tools": tools });
    let attach = crate::context::prepare_attachments(state, &synthetic).await;
    let mode = parse_model_mode(model_name);
    let resolved = state.resolve_model(&mode.base_model).await;
    let profile = client_profiles::detect_profile(headers, &synthetic);
    let built = prompt_builder::messages_to_prompt(&synthetic, &profile, &attach.inline_note);
    let stream = false; // 由路由 action 決定，呼叫端覆寫
    StandardRequest {
        response_model: model_name.to_string(),
        resolved_model: resolved,
        prompt: built.prompt,
        stream,
        thinking_enabled: if mode.force_thinking { Some(true) } else { None },
        force_thinking: mode.force_thinking,
        enable_search: mode.chat_type == "deep_research",
        chat_type: mode.chat_type,
        tools: built.tools,
        tool_names: built.tool_names,
        surface: "gemini".to_string(),
        image_options: None,
        max_tokens: None,
        client_profile: profile,
        files: attach.files,
        bound_account: attach.bound_account,
        caller: None,
        exclude_accounts: Default::default(),
    }
}

/// OpenAI Chat Completions / 通用 chat 的標準請求。
pub async fn build_openai_request(
    state: &AppState,
    body: &Value,
    headers: &HeaderMap,
    surface: &str,
    default_model: &str,
) -> StandardRequest {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(default_model)
        .to_string();
    let mode = parse_model_mode(&model_name);
    let resolved = state.resolve_model(&mode.base_model).await;
    let profile = client_profiles::detect_profile(headers, body);
    let attach = crate::context::prepare_attachments(state, body).await;
    let built = prompt_builder::messages_to_prompt(body, &profile, &attach.inline_note);

    // 影像/影片生成：用純粹的使用者描述文字當 prompt（避免 Human/Assistant 包裝干擾出圖）
    let prompt = if mode.chat_type == "t2i" || mode.chat_type == "t2v" {
        last_user_text(body).unwrap_or(built.prompt)
    } else {
        built.prompt
    };

    let (mut thinking_enabled, _forced) = extract_thinking(body, mode.force_thinking);
    let enable_search =
        mode.chat_type == "deep_research" || coerce_bool(body.get("enable_search")).unwrap_or(false);
    let stream = coerce_bool(body.get("stream")).unwrap_or(false);
    if surface == "anthropic"
        && stream
        && mode.chat_type == "t2t"
        && profile == client_profiles::CLAUDE_CODE
    {
        thinking_enabled = Some(true);
    }
    let max_tokens = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|v| v.as_i64());

    StandardRequest {
        response_model: model_name,
        resolved_model: resolved,
        prompt,
        stream,
        thinking_enabled,
        force_thinking: mode.force_thinking,
        enable_search,
        chat_type: mode.chat_type,
        tools: built.tools,
        tool_names: built.tool_names,
        surface: surface.to_string(),
        image_options: None as Option<ImageOptions>,
        max_tokens,
        client_profile: profile,
        files: attach.files,
        bound_account: attach.bound_account,
        caller: None,
        exclude_accounts: Default::default(),
    }
}
