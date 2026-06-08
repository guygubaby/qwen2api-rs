//! OpenAI chat.completion.chunk 串流翻譯器，對應 Python `services/openai_stream_translator.py`。

use super::{OutEvent, Usage};
use crate::toolcall::ParsedToolCall;
use crate::util::{now_unix, short_id};
use serde_json::{json, Value};

pub struct OpenAiStreamTranslator {
    pub id: String,
    pub created: i64,
    pub model: String,
    role_sent: bool,
}

impl OpenAiStreamTranslator {
    pub fn new(model: &str) -> Self {
        OpenAiStreamTranslator {
            id: format!("chatcmpl-{}", short_id(12)),
            created: now_unix(),
            model: model.to_string(),
            role_sent: false,
        }
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> String {
        let v = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        });
        format!("data: {}\n\n", v)
    }

    /// 將一個 OutEvent 轉成 0..N 個 SSE 字串。
    pub fn on_event(&mut self, ev: &OutEvent) -> Vec<String> {
        let mut out = Vec::new();
        match ev {
            OutEvent::ReasoningDelta(r) => {
                let mut delta = json!({ "reasoning_content": r });
                if !self.role_sent {
                    delta["role"] = json!("assistant");
                    self.role_sent = true;
                }
                out.push(self.chunk(delta, None));
            }
            OutEvent::ContentDelta(c) => {
                let mut delta = json!({ "content": c });
                if !self.role_sent {
                    delta["role"] = json!("assistant");
                    self.role_sent = true;
                }
                out.push(self.chunk(delta, None));
            }
            OutEvent::ToolCalls(tcs) => {
                let tool_calls: Vec<Value> = tcs
                    .iter()
                    .enumerate()
                    .map(|(i, tc)| tool_call_chunk(i, tc))
                    .collect();
                let mut delta = json!({ "tool_calls": tool_calls });
                if !self.role_sent {
                    delta["role"] = json!("assistant");
                    self.role_sent = true;
                }
                out.push(self.chunk(delta, None));
            }
            OutEvent::Done { usage, finish_reason, .. } => {
                // 最終 chunk：空 delta + finish_reason + usage
                let v = json!({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                    "usage": usage_json(usage),
                });
                out.push(format!("data: {}\n\n", v));
                out.push("data: [DONE]\n\n".to_string());
            }
            OutEvent::Error(e) => {
                let v = json!({ "error": { "message": e, "type": "upstream_error" } });
                out.push(format!("data: {}\n\n", v));
                out.push("data: [DONE]\n\n".to_string());
            }
        }
        out
    }
}

pub fn tool_call_chunk(index: usize, tc: &ParsedToolCall) -> Value {
    json!({
        "index": index,
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into()),
        },
    })
}

pub fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "completion_tokens_details": { "reasoning_tokens": u.reasoning_tokens },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolcall::ParsedToolCall;
    use serde_json::json;

    /// 把 translator 產生的 `"data: <json>\n\n"` 字串還原成 Value，方便斷言。
    fn parse_chunk_payload(s: &str) -> Value {
        let body = s.trim_start_matches("data: ").trim();
        serde_json::from_str(body).unwrap_or_else(|e| panic!("failed to parse chunk JSON: {e}\nraw: {s}"))
    }

    /// role:"assistant" 只在第一個 delta 出現一次（OpenAI 串流契約）。
    #[test]
    fn first_content_delta_includes_role_subsequent_do_not() {
        let mut t = OpenAiStreamTranslator::new("gpt-4o");
        let first = t.on_event(&OutEvent::ContentDelta("hi".into()));
        assert_eq!(first.len(), 1);
        let v1 = parse_chunk_payload(&first[0]);
        assert_eq!(v1["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v1["choices"][0]["delta"]["content"], "hi");
        assert_eq!(v1["object"], "chat.completion.chunk");
        assert_eq!(v1["model"], "gpt-4o");

        let second = t.on_event(&OutEvent::ContentDelta(" there".into()));
        let v2 = parse_chunk_payload(&second[0]);
        assert!(v2["choices"][0]["delta"].get("role").is_none(), "role should be sent only once");
        assert_eq!(v2["choices"][0]["delta"]["content"], " there");
    }

    /// reasoning 走 `reasoning_content` 欄位；首次也帶 role，之後 ContentDelta 不再帶。
    #[test]
    fn reasoning_first_sends_role_then_content_does_not_repeat() {
        let mut t = OpenAiStreamTranslator::new("gpt-4o");
        let chunks = t.on_event(&OutEvent::ReasoningDelta("rethink".into()));
        let v = parse_chunk_payload(&chunks[0]);
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "rethink");

        let next = t.on_event(&OutEvent::ContentDelta("body".into()));
        let vn = parse_chunk_payload(&next[0]);
        assert!(vn["choices"][0]["delta"].get("role").is_none(), "role should not be repeated");
    }

    /// Done：產 2 條（最終 chunk 含 usage + finish_reason，加上 `[DONE]` 結尾）。
    #[test]
    fn done_emits_final_chunk_with_usage_then_done_sentinel() {
        let mut t = OpenAiStreamTranslator::new("gpt-4o");
        let usage = Usage { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8, reasoning_tokens: 1 };
        let out = t.on_event(&OutEvent::Done { usage, finish_reason: "stop".into(), email: None });
        assert_eq!(out.len(), 2, "Done should emit final chunk plus [DONE]");
        assert_eq!(out[1], "data: [DONE]\n\n");

        let final_chunk = parse_chunk_payload(&out[0]);
        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(final_chunk["usage"]["prompt_tokens"], 3);
        assert_eq!(final_chunk["usage"]["completion_tokens"], 5);
        assert_eq!(final_chunk["usage"]["total_tokens"], 8);
        assert_eq!(final_chunk["usage"]["completion_tokens_details"]["reasoning_tokens"], 1);
    }

    /// Error：產錯誤 chunk + [DONE]，符合 OpenAI 錯誤串流慣例。
    #[test]
    fn error_event_emits_error_chunk_and_done() {
        let mut t = OpenAiStreamTranslator::new("gpt-4o");
        let out = t.on_event(&OutEvent::Error("upstream 502".into()));
        assert_eq!(out.len(), 2);
        let err = parse_chunk_payload(&out[0]);
        assert_eq!(err["error"]["message"], "upstream 502");
        assert_eq!(err["error"]["type"], "upstream_error");
        assert_eq!(out[1], "data: [DONE]\n\n");
    }

    /// ToolCalls：欄位佈局符合 OpenAI（index / id / type=function / function.{name,arguments-as-string}）。
    #[test]
    fn tool_calls_event_lays_out_openai_shape() {
        let mut t = OpenAiStreamTranslator::new("gpt-4o");
        let tc = ParsedToolCall {
            id: "call_xyz".into(),
            name: "Bash".into(),
            arguments: json!({ "cmd": "ls", "n": 2 }),
        };
        let chunks = t.on_event(&OutEvent::ToolCalls(vec![tc]));
        let v = parse_chunk_payload(&chunks[0]);
        let item = &v["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(item["index"], 0);
        assert_eq!(item["id"], "call_xyz");
        assert_eq!(item["type"], "function");
        assert_eq!(item["function"]["name"], "Bash");
        // arguments 須是 JSON 字串（OpenAI 規範），而非物件
        let args_str = item["function"]["arguments"].as_str().expect("arguments should be a string");
        let args: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(args["cmd"], "ls");
        assert_eq!(args["n"], 2);
    }

    /// usage_json 欄位佈局穩定（前端/客戶端依賴 completion_tokens_details.reasoning_tokens 取思考 token）。
    #[test]
    fn usage_json_layout_is_stable() {
        let u = Usage { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3, reasoning_tokens: 4 };
        let v = usage_json(&u);
        assert_eq!(v["prompt_tokens"], 1);
        assert_eq!(v["completion_tokens"], 2);
        assert_eq!(v["total_tokens"], 3);
        assert_eq!(v["completion_tokens_details"]["reasoning_tokens"], 4);
    }
}
