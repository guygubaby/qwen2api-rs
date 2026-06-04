//! 解析 Qwen 上游 SSE，對應 Python `upstream/sse_consumer.py`（並擴充 qwen3.7 的 summary_thought 格式）。

use serde_json::Value;

/// 一個正規化後的上游 delta。
#[derive(Debug, Clone, Default)]
pub struct QwenDelta {
    pub phase: String,
    /// 原始 delta.content（answer 階段為增量文字）。
    pub content: String,
    /// 思考內容（累積）：qwen3.7 的 extra.summary_thought.content join 後字串。
    pub reasoning_cumulative: Option<String>,
    /// 思考內容（增量）：舊格式 reasoning_content/reasoning/thinking 等。
    pub reasoning_incremental: String,
    pub status: String,
    /// 該事件的 usage（若有）。
    pub usage: Option<Value>,
}

fn first_text(values: &[Option<&Value>]) -> String {
    for v in values {
        if let Some(Value::String(s)) = v {
            if !s.is_empty() {
                return s.clone();
            }
        }
    }
    String::new()
}

/// 從 delta 抓舊格式增量 reasoning。
fn extract_reasoning_incremental(delta: &Value) -> String {
    let extra = delta.get("extra");
    fn ge<'a>(obj: Option<&'a Value>, key: &str) -> Option<&'a Value> {
        obj.and_then(|o| o.get(key))
    }
    first_text(&[
        delta.get("reasoning_content"),
        delta.get("reasoning"),
        delta.get("reasoning_text"),
        delta.get("thinking"),
        delta.get("thoughts"),
        ge(extra, "reasoning_content"),
        ge(extra, "reasoning"),
        ge(extra, "reasoning_text"),
        ge(extra, "thinking"),
        ge(extra, "thoughts"),
    ])
}

/// 從 extra.summary_thought.content（陣列）join 出累積思考文字。
fn extract_reasoning_cumulative(delta: &Value) -> Option<String> {
    let arr = delta
        .get("extra")?
        .get("summary_thought")?
        .get("content")?
        .as_array()?;
    let joined: String = arr
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// 解析一個 SSE 訊息塊（可能含多行 data:）。回傳正規化 delta 列表。
pub fn parse_sse_chunk(chunk: &str) -> Vec<QwenDelta> {
    let mut out = Vec::new();
    for raw_line in chunk.lines() {
        let line = raw_line.trim();
        if !line.starts_with("data:") {
            continue;
        }
        let data = line[5..].trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let obj: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // 首事件 response.created 沒有 choices
        let choices = match obj.get("choices").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };
        let delta = match choices[0].get("delta") {
            Some(d) => d,
            None => continue,
        };
        let phase = delta.get("phase").and_then(|v| v.as_str()).unwrap_or("answer").to_string();
        let content = delta.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let status = delta.get("status").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let reasoning_cumulative = extract_reasoning_cumulative(delta);
        let reasoning_incremental = extract_reasoning_incremental(delta);
        let usage = obj.get("usage").cloned();
        out.push(QwenDelta {
            phase,
            content,
            reasoning_cumulative,
            reasoning_incremental,
            status,
            usage,
        });
    }
    out
}

/// 偵測上游明確 JSON 錯誤（{"success":false} 或 {"error":...}），回錯誤訊息。
pub fn extract_upstream_error(text: &str) -> Option<String> {
    for raw_line in text.lines() {
        let mut line = raw_line.trim();
        if line.starts_with("data:") {
            line = line[5..].trim();
        }
        if line.is_empty() || line == "[DONE]" || !line.starts_with('{') {
            continue;
        }
        let obj: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(msg) = format_upstream_error(&obj) {
            return Some(msg);
        }
    }
    None
}

/// delta 是否含「有意義」的內容（content / reasoning 任一非空）。
/// 用於 executor 偵測「空回覆」並觸發跨帳號重試——上游偶發回完整 SSE 但 0 字內容，
/// 對 client 看起來就是任務莫名停下。
pub fn delta_has_meaningful_content(d: &QwenDelta) -> bool {
    !d.content.is_empty()
        || !d.reasoning_incremental.is_empty()
        || d.reasoning_cumulative.as_deref().map_or(false, |s| !s.is_empty())
}

/// delta 是否含「真正的回覆內容」（answer-phase content；不含 reasoning/thinking）。
/// 用於 executor 判定「客戶端是否已看到實際回答」——
/// thinking 模型若只跑完思考就斷掉（quota 中斷、上游無 answer），對 Claude/OpenAI/Gemini
/// 客戶端等於沒收到任何答覆，此時應跨帳號重試，避免被誤判為成功。
/// phase 預設值同 mod.rs：未設時當作 "answer"，所以非思考階段也計入。
pub fn delta_has_answer_content(d: &QwenDelta) -> bool {
    !d.content.is_empty() && d.phase != "think" && d.phase != "thinking_summary"
}

fn format_upstream_error(obj: &Value) -> Option<String> {
    let request_id = obj
        .get("request_id")
        .or_else(|| obj.get("response_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("-");
    if obj.get("success") == Some(&Value::Bool(false)) {
        let data = obj.get("data");
        let code = data
            .and_then(|d| d.get("code"))
            .or_else(|| obj.get("code"))
            .and_then(|v| v.as_str())
            .unwrap_or("upstream_error");
        let details = data
            .and_then(|d| d.get("details").or_else(|| d.get("message")))
            .or_else(|| obj.get("details"))
            .or_else(|| obj.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        return Some(format!("Qwen upstream error code={code} request_id={request_id} details={details}"));
    }
    if let Some(err) = obj.get("error") {
        if let Some(eo) = err.as_object() {
            let code = eo.get("code").and_then(|v| v.as_str()).unwrap_or("upstream_error");
            let details = eo
                .get("details")
                .or_else(|| eo.get("message"))
                .or_else(|| eo.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            return Some(format!("Qwen upstream error code={code} request_id={request_id} details={details}"));
        }
        if let Some(s) = err.as_str() {
            if !s.is_empty() {
                return Some(format!("Qwen upstream error request_id={request_id} details={s}"));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chunk_of(payload: serde_json::Value) -> String {
        format!("data: {}\n\n", payload)
    }

    /// 空輸入、純 [DONE]、非 data: 行 → 一律不產 delta。
    #[test]
    fn empty_done_and_non_data_lines_yield_nothing() {
        assert!(parse_sse_chunk("").is_empty());
        assert!(parse_sse_chunk("data: [DONE]\n\n").is_empty());
        assert!(parse_sse_chunk("data: \n\n").is_empty());
        assert!(parse_sse_chunk("event: ping\n\n").is_empty());
    }

    /// 首事件 `response.created` 無 choices、或 choices 空、或缺 delta → 全跳過。
    #[test]
    fn payloads_without_choices_or_delta_are_skipped() {
        let head = chunk_of(json!({ "response_id": "x", "object": "response.created" }));
        let empty = chunk_of(json!({ "choices": [] }));
        let no_delta = chunk_of(json!({ "choices": [{}] }));
        assert!(parse_sse_chunk(&head).is_empty());
        assert!(parse_sse_chunk(&empty).is_empty());
        assert!(parse_sse_chunk(&no_delta).is_empty());
    }

    /// answer 階段：phase、content、status 直取；reasoning 兩欄皆空。
    #[test]
    fn answer_delta_extracts_phase_content_status() {
        let c = chunk_of(json!({
            "choices": [{ "delta": { "phase": "answer", "content": "hi", "status": "in_progress" } }]
        }));
        let v = parse_sse_chunk(&c);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].phase, "answer");
        assert_eq!(v[0].content, "hi");
        assert_eq!(v[0].status, "in_progress");
        assert_eq!(v[0].reasoning_incremental, "");
        assert!(v[0].reasoning_cumulative.is_none());
    }

    /// 舊格式 reasoning：reasoning_content / reasoning / thinking / extra.* 任一非空都會被抓。
    #[test]
    fn legacy_reasoning_keys_are_picked_up() {
        let c = chunk_of(json!({
            "choices": [{ "delta": { "phase": "think", "reasoning_content": "thought-a" } }]
        }));
        assert_eq!(parse_sse_chunk(&c)[0].reasoning_incremental, "thought-a");

        // extra 內嵌的 thinking 也算
        let c2 = chunk_of(json!({
            "choices": [{ "delta": { "extra": { "thinking": "deep" } } }]
        }));
        assert_eq!(parse_sse_chunk(&c2)[0].reasoning_incremental, "deep");
    }

    /// qwen3.7 summary_thought：以 `\n\n` join 整個陣列 → reasoning_cumulative。
    #[test]
    fn qwen37_summary_thought_joins_with_double_newline() {
        let c = chunk_of(json!({
            "choices": [{ "delta": {
                "extra": { "summary_thought": { "content": ["a", "b", "c"] } }
            } }]
        }));
        assert_eq!(parse_sse_chunk(&c)[0].reasoning_cumulative.as_deref(), Some("a\n\nb\n\nc"));
    }

    /// summary_thought.content 為空陣列 → cumulative 為 None（避免送空字串）。
    #[test]
    fn empty_summary_thought_yields_none_cumulative() {
        let c = chunk_of(json!({
            "choices": [{ "delta": {
                "extra": { "summary_thought": { "content": [] } }
            } }]
        }));
        assert!(parse_sse_chunk(&c)[0].reasoning_cumulative.is_none());
    }

    /// 上游 usage 整塊原樣帶出。
    #[test]
    fn usage_field_passed_through() {
        let c = chunk_of(json!({
            "choices": [{ "delta": { "content": "" } }],
            "usage": { "prompt_tokens": 7, "completion_tokens": 11 }
        }));
        let v = parse_sse_chunk(&c);
        let usage = v[0].usage.as_ref().expect("usage 必有");
        assert_eq!(usage.get("prompt_tokens").and_then(|v| v.as_i64()), Some(7));
        assert_eq!(usage.get("completion_tokens").and_then(|v| v.as_i64()), Some(11));
    }

    /// 同一 chunk 內多個 data: 行 → 各自成一個 QwenDelta。
    #[test]
    fn multi_data_lines_yield_multiple_deltas() {
        let a = chunk_of(json!({ "choices": [{ "delta": { "content": "x" } }] }));
        let b = chunk_of(json!({ "choices": [{ "delta": { "content": "y" } }] }));
        let v = parse_sse_chunk(&format!("{a}{b}"));
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].content, "x");
        assert_eq!(v[1].content, "y");
    }

    /// 損壞 JSON 不應中斷解析，只是跳過該行。
    #[test]
    fn malformed_json_is_skipped_not_panicking() {
        let mixed = "data: {not json}\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n";
        let v = parse_sse_chunk(mixed);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].content, "ok");
    }

    /// 錯誤格式 1：`{"success":false, data:{code, details}}` → 格式化錯誤訊息含三者。
    #[test]
    fn extract_error_handles_success_false() {
        let s = r#"data: {"success":false,"request_id":"req-1","data":{"code":"rate_limit","details":"too fast"}}"#;
        let err = extract_upstream_error(s).expect("應偵測到錯誤");
        assert!(err.contains("rate_limit"), "缺 code: {err}");
        assert!(err.contains("req-1"), "缺 request_id: {err}");
        assert!(err.contains("too fast"), "缺 details: {err}");
    }

    /// 錯誤格式 2：`{"error":{code, message}}` → 同樣格式化。
    #[test]
    fn extract_error_handles_error_object() {
        let s = r#"data: {"error":{"code":"auth_error","message":"invalid token"}}"#;
        let err = extract_upstream_error(s).expect("應偵測到錯誤");
        assert!(err.contains("auth_error"));
        assert!(err.contains("invalid token"));
    }

    /// delta_has_meaningful_content：判斷一筆 delta 是否含 content/reasoning 任一非空。
    /// 用於空回覆重試判定（executor 不能把完全空的 delta 序列當 success）。
    #[test]
    fn meaningful_content_detection() {
        let empty = QwenDelta::default();
        assert!(!delta_has_meaningful_content(&empty), "空 delta 應為 false");

        let only_phase = QwenDelta { phase: "answer".into(), ..Default::default() };
        assert!(!delta_has_meaningful_content(&only_phase), "只有 phase 沒文字應為 false");

        let with_content = QwenDelta { content: "hi".into(), ..Default::default() };
        assert!(delta_has_meaningful_content(&with_content));

        let with_inc_reasoning = QwenDelta { reasoning_incremental: "thinking".into(), ..Default::default() };
        assert!(delta_has_meaningful_content(&with_inc_reasoning));

        let with_cumul_reasoning = QwenDelta { reasoning_cumulative: Some("acc".into()), ..Default::default() };
        assert!(delta_has_meaningful_content(&with_cumul_reasoning));

        // cumulative=Some("") 也應視為空（防止上游送 "" 仍被誤判為有內容）
        let empty_cumul = QwenDelta { reasoning_cumulative: Some("".into()), ..Default::default() };
        assert!(!delta_has_meaningful_content(&empty_cumul));
    }

    /// delta_has_answer_content：只把 answer 階段的 content 算「真正回覆」，
    /// 思考階段（think / thinking_summary）即便 content 非空也不算（罕見，但上游有這種事件）。
    /// 用於 executor 判定「客戶端是否已看到實際答案」，避免 thinking 中斷被誤判成功。
    #[test]
    fn answer_content_detection_excludes_thinking_phase() {
        // 空 delta、純 reasoning：皆非答覆。
        assert!(!delta_has_answer_content(&QwenDelta::default()));
        assert!(!delta_has_answer_content(&QwenDelta {
            reasoning_incremental: "thinking".into(),
            ..Default::default()
        }));
        assert!(!delta_has_answer_content(&QwenDelta {
            reasoning_cumulative: Some("acc".into()),
            ..Default::default()
        }));

        // answer 階段的 content → 算答覆（phase 預設值與 mod.rs 一致為 "answer"）。
        assert!(delta_has_answer_content(&QwenDelta {
            phase: "answer".into(),
            content: "hi".into(),
            ..Default::default()
        }));
        // 即使 phase 沒帶（解析器預設給 "answer"），content 非空仍算答覆。
        assert!(delta_has_answer_content(&QwenDelta { content: "hi".into(), ..Default::default() }));

        // think / thinking_summary 階段的 content 不算答覆（避免被誤當成「已回應」）。
        assert!(!delta_has_answer_content(&QwenDelta {
            phase: "think".into(),
            content: "thought-as-content".into(),
            ..Default::default()
        }));
        assert!(!delta_has_answer_content(&QwenDelta {
            phase: "thinking_summary".into(),
            content: "summary-as-content".into(),
            ..Default::default()
        }));
    }

    /// 正常 SSE chunk 與 [DONE] 都不會被誤判為錯誤。
    #[test]
    fn extract_error_ignores_normal_payload() {
        let s = r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#;
        assert!(extract_upstream_error(s).is_none());
        assert!(extract_upstream_error("data: [DONE]\n\n").is_none());
    }
}
