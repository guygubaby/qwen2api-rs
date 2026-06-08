//! 串流 SSE 字串格式器，對應 Python `runtime/stream_presenter.py`。
//! 提供 Anthropic 事件流與 Gemini chunk。

use super::{OutEvent, Usage};
use crate::util::{now_unix, short_id};
use serde_json::{json, Value};

const UNSIGNED_THINKING_SIGNATURE: &str = "cXdlbjJhcGktcnMtdW5zaWduZWQtdGhpbmtpbmc=";

/// SSE event 行（Anthropic 用 event: + data:）。
fn sse_event(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

#[derive(PartialEq)]
enum Block {
    None,
    Thinking,
    Text,
}

/// Anthropic Messages 串流翻譯器。
pub struct AnthropicStreamTranslator {
    pub id: String,
    pub model: String,
    prompt_tokens: i64,
    current: Block,
    index: i64,
    started: bool,
}

impl AnthropicStreamTranslator {
    pub fn new(model: &str, prompt_tokens: i64) -> Self {
        AnthropicStreamTranslator {
            id: format!("msg_{}", short_id(12)),
            model: model.to_string(),
            prompt_tokens,
            current: Block::None,
            index: 0,
            started: false,
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        let msg = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": { "input_tokens": self.prompt_tokens, "output_tokens": 0 },
            }
        });
        out.push(sse_event("message_start", &msg));
    }

    fn close_block(&mut self, out: &mut Vec<String>) {
        if self.current != Block::None {
            if self.current == Block::Thinking {
                out.push(sse_event(
                    "content_block_delta",
                    &json!({"type":"content_block_delta","index": self.index,
                            "delta": {"type":"signature_delta","signature": UNSIGNED_THINKING_SIGNATURE}}),
                ));
            }
            out.push(sse_event("content_block_stop", &json!({"type":"content_block_stop","index": self.index})));
            self.index += 1;
            self.current = Block::None;
        }
    }

    fn open_block(&mut self, out: &mut Vec<String>, kind: Block) {
        let (btype, cb) = match kind {
            Block::Thinking => ("thinking", json!({"type":"thinking","thinking":"","signature":UNSIGNED_THINKING_SIGNATURE})),
            Block::Text => ("text", json!({"type":"text","text":""})),
            Block::None => return,
        };
        let _ = btype;
        out.push(sse_event(
            "content_block_start",
            &json!({"type":"content_block_start","index": self.index, "content_block": cb}),
        ));
        self.current = kind;
    }

    pub fn on_event(&mut self, ev: &OutEvent) -> Vec<String> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);
        match ev {
            OutEvent::ReasoningDelta(r) => {
                if self.current != Block::Thinking {
                    self.close_block(&mut out);
                    self.open_block(&mut out, Block::Thinking);
                }
                out.push(sse_event(
                    "content_block_delta",
                    &json!({"type":"content_block_delta","index": self.index,
                            "delta": {"type":"thinking_delta","thinking": r}}),
                ));
            }
            OutEvent::ContentDelta(c) => {
                if self.current != Block::Text {
                    self.close_block(&mut out);
                    self.open_block(&mut out, Block::Text);
                }
                out.push(sse_event(
                    "content_block_delta",
                    &json!({"type":"content_block_delta","index": self.index,
                            "delta": {"type":"text_delta","text": c}}),
                ));
            }
            OutEvent::ToolCalls(tcs) => {
                self.close_block(&mut out);
                for tc in tcs {
                    out.push(sse_event(
                        "content_block_start",
                        &json!({"type":"content_block_start","index": self.index,
                                "content_block": {"type":"tool_use","id": tc.id, "name": tc.name, "input": {}}}),
                    ));
                    let pj = serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into());
                    out.push(sse_event(
                        "content_block_delta",
                        &json!({"type":"content_block_delta","index": self.index,
                                "delta": {"type":"input_json_delta","partial_json": pj}}),
                    ));
                    out.push(sse_event("content_block_stop", &json!({"type":"content_block_stop","index": self.index})));
                    self.index += 1;
                }
                self.current = Block::None;
            }
            OutEvent::Done { usage, finish_reason, .. } => {
                self.close_block(&mut out);
                let stop_reason = if finish_reason == "tool_calls" { "tool_use" } else { "end_turn" };
                out.push(sse_event(
                    "message_delta",
                    &json!({"type":"message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                            "usage": {"output_tokens": usage.completion_tokens}}),
                ));
                out.push(sse_event("message_stop", &json!({"type":"message_stop"})));
            }
            OutEvent::Error(e) => {
                // 先關閉開啟中的 content_block，再送 error 與 message_stop，保持事件序列合法
                self.close_block(&mut out);
                out.push(sse_event(
                    "error",
                    &json!({"type":"error","error":{"type":"upstream_error","message": e}}),
                ));
                out.push(sse_event("message_stop", &json!({"type":"message_stop"})));
            }
        }
        out
    }
}

/// Gemini 串流：每個內容 delta 輸出一個 JSON chunk（陣列元素風格，NDJSON）。
pub fn gemini_text_chunk(text: &str) -> String {
    let v = json!({
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "index": 0,
        }]
    });
    format!("data: {}\n\n", v)
}

pub fn gemini_final_chunk(usage: &Usage) -> String {
    let v = json!({
        "candidates": [{
            "content": {"parts": [{"text": ""}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.prompt_tokens,
            "candidatesTokenCount": usage.completion_tokens,
            "totalTokenCount": usage.total_tokens,
        }
    });
    format!("data: {}\n\n", v)
}

pub fn gemini_error_chunk(msg: &str) -> String {
    let v = json!({ "error": { "message": msg } });
    format!("data: {}\n\n", v)
}

/// 通用 OpenAI 風格 error SSE。
pub fn openai_error_sse(msg: &str) -> String {
    let v = json!({ "error": { "message": msg, "type": "upstream_error" } });
    format!("data: {}\n\ndata: [DONE]\n\n", v)
}

/// 給未使用的 short_id/now_unix 保活（避免 dead_code 警告於某些編譯路徑）。
#[allow(dead_code)]
fn _keepalive() -> (String, i64) {
    (short_id(4), now_unix())
}
