//! 構建 POST /api/v2/chat/completions 的請求 body，對應 Python `upstream/payload_builder.py`。

use crate::util::{now_unix, uuid4};
use serde_json::{json, Value};

/// 影像生成選項。
#[derive(Debug, Clone, Default)]
pub struct ImageOptions {
    pub size: Option<String>,
    pub ratio: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct BuildPayloadArgs<'a> {
    pub chat_id: &'a str,
    pub model: &'a str,
    pub content: &'a str,
    pub has_custom_tools: bool,
    pub files: Vec<Value>,
    pub chat_type: &'a str,
    pub image_options: Option<ImageOptions>,
    pub thinking_enabled: Option<bool>,
    pub enable_search: bool,
}

fn apply_thinking(fc: &mut serde_json::Map<String, Value>, enabled: bool) {
    fc.insert("thinking_enabled".into(), json!(enabled));
    fc.insert("auto_thinking".into(), json!(enabled));
    fc.insert("thinking_mode".into(), json!(if enabled { "Auto" } else { "Disabled" }));
}

pub fn build_chat_payload(args: &BuildPayloadArgs) -> Value {
    let ts = now_unix();
    let is_image_gen = matches!(args.chat_type, "image_gen" | "t2i");
    let is_video_gen = args.chat_type == "t2v";
    let img = args.image_options.clone().unwrap_or_default();

    // 基礎 feature_config
    let mut fc = serde_json::Map::new();
    fc.insert("thinking_enabled".into(), json!(true));
    fc.insert("output_schema".into(), json!("phase"));
    fc.insert("research_mode".into(), json!("normal"));
    fc.insert("auto_thinking".into(), json!(true));
    fc.insert("thinking_mode".into(), json!("Auto"));
    fc.insert("thinking_format".into(), json!("summary"));
    fc.insert("code_interpreter".into(), json!(false));
    // 關鍵：強制關閉原生工具，避免上游攔截本地文字工具指令
    fc.insert("function_calling".into(), json!(false));
    fc.insert("enable_tools".into(), json!(false));
    fc.insert("enable_function_call".into(), json!(false));
    fc.insert("tool_choice".into(), json!("none"));
    fc.insert("auto_search".into(), json!(args.enable_search || args.chat_type == "deep_research"));
    fc.insert("plugins_enabled".into(), json!(is_image_gen));
    fc.insert("image_gen".into(), json!(is_image_gen));
    fc.insert("image_generation".into(), json!(is_image_gen));

    if let Some(t) = args.thinking_enabled {
        apply_thinking(&mut fc, t);
    }
    if is_image_gen || is_video_gen {
        apply_thinking(&mut fc, false);
    }
    if is_image_gen {
        if let Some(v) = &img.size {
            fc.insert("image_size".into(), json!(v));
        }
        if let Some(v) = &img.ratio {
            fc.insert("image_ratio".into(), json!(v));
            fc.insert("aspect_ratio".into(), json!(v));
        }
        if let Some(v) = img.width {
            fc.insert("width".into(), json!(v));
        }
        if let Some(v) = img.height {
            fc.insert("height".into(), json!(v));
        }
    }

    // extra.meta
    let mut meta = serde_json::Map::new();
    meta.insert("subChatType".into(), json!(args.chat_type));
    if is_image_gen {
        if let Some(v) = &img.size {
            meta.insert("imageSize".into(), json!(v));
        }
        if let Some(v) = &img.ratio {
            meta.insert("imageRatio".into(), json!(v));
            meta.insert("aspectRatio".into(), json!(v));
        }
        if let Some(v) = img.width {
            meta.insert("width".into(), json!(v));
        }
        if let Some(v) = img.height {
            meta.insert("height".into(), json!(v));
        }
    }

    let mut payload = json!({
        "stream": true,
        "version": "2.1",
        "incremental_output": true,
        "chat_id": args.chat_id,
        "chat_mode": "normal",
        "model": args.model,
        "parent_id": Value::Null,
        "messages": [{
            "fid": uuid4(),
            "parentId": Value::Null,
            "childrenIds": [uuid4()],
            "role": "user",
            "content": args.content,
            "user_action": "chat",
            "files": args.files,
            "timestamp": ts,
            "models": [args.model],
            "chat_type": args.chat_type,
            "feature_config": Value::Object(fc),
            "extra": {"meta": Value::Object(meta)},
            "sub_chat_type": args.chat_type,
            "parent_id": Value::Null,
        }],
        "timestamp": ts,
    });

    // 圖片(t2i)/影片(t2v)：實測上游用 `size` = 比例字串（如 "1:1"），同時放訊息層與頂層。
    if is_image_gen || is_video_gen {
        let size_str = img.ratio.clone().unwrap_or_else(|| "1:1".to_string());
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("size".into(), json!(size_str));
            if let Some(msgs) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
                if let Some(msg0) = msgs.get_mut(0).and_then(|m| m.as_object_mut()) {
                    msg0.insert("size".into(), json!(size_str));
                }
            }
        }
    }

    payload
}
