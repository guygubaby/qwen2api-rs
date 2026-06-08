//! 阿里雲 OSS V4 簽名 + Qwen 檔案上傳流程，對應 Python `services/upstream_file_uploader.py`（用 oss2）。
//! 簽名演算法逐行對齊 oss2 2.19.1 的 ProviderAuthV4（見 dev/PROTOCOL.md）。

use crate::error::{AppError, AppResult};
use crate::upstream::client::{BASE_URL, UA};
use crate::util::{now_millis, utc_iso8601_basic, uuid4};
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Duration;

type HmacSha256 = Hmac<Sha256>;

/// oss2 的 __v4_uri_encode：保留 A-Za-z0-9 _-~. ；ignore_slashes 時保留 '/'；其餘 %XX 大寫。
fn v4_uri_encode(raw: &str, ignore_slashes: bool) -> String {
    let mut res = String::with_capacity(raw.len());
    for &b in raw.as_bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '~' | '.') {
            res.push(c);
        } else if ignore_slashes && c == '/' {
            res.push('/');
        } else {
            res.push_str(&format!("%{:02X}", b));
        }
    }
    res
}

fn hmac(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("hmac key");
    m.update(msg);
    m.finalize().into_bytes().to_vec()
}

struct StsCreds<'a> {
    ak: &'a str,
    sk: &'a str,
    token: &'a str,
    region: &'a str, // 已去除 oss- 前綴
}

/// 對 OSS PUT 請求做 V4 簽名，回傳 (authorization, x-oss-date, x-oss-content-sha256)。
fn sign_put(creds: &StsCreds, bucket: &str, key: &str, content_type: &str) -> (String, String, String) {
    let (xdate, ymd) = utc_iso8601_basic();
    let payload_hash = "UNSIGNED-PAYLOAD";

    // 簽名 headers：content-type, x-oss-content-sha256, x-oss-date, x-oss-security-token（已字典序）
    let canonical_headers = format!(
        "content-type:{}\nx-oss-content-sha256:{}\nx-oss-date:{}\nx-oss-security-token:{}\n",
        content_type, payload_hash, xdate, creds.token
    );
    let canonical_uri = v4_uri_encode(&format!("/{}/{}", bucket, key), true);
    let canonical_request = format!(
        "PUT\n{}\n{}\n{}\n{}\n{}",
        canonical_uri, /*query*/ "", canonical_headers, /*additional*/ "", payload_hash
    );
    let scope = format!("{}/{}/oss/aliyun_v4_request", ymd, creds.region);
    let cr_hash = hex::encode(Sha256::digest(canonical_request.as_bytes()));
    let string_to_sign = format!("OSS4-HMAC-SHA256\n{}\n{}\n{}", xdate, scope, cr_hash);

    // signing key
    let k_secret = format!("aliyun_v4{}", creds.sk);
    let k_date = hmac(k_secret.as_bytes(), ymd.as_bytes());
    let k_region = hmac(&k_date, creds.region.as_bytes());
    let k_product = hmac(&k_region, b"oss");
    let k_signing = hmac(&k_product, b"aliyun_v4_request");
    let signature = hex::encode(hmac(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "OSS4-HMAC-SHA256 Credential={}/{}, Signature={}",
        creds.ak, scope, signature
    );
    (authorization, xdate, payload_hash.to_string())
}

fn file_class(content_type: &str) -> &'static str {
    if content_type.starts_with("image/") {
        "image"
    } else if content_type.starts_with("audio/") {
        "audio"
    } else if content_type.starts_with("video/") {
        "video"
    } else {
        "document"
    }
}

/// 上傳本地檔案 bytes 到 Qwen 上游，回傳可放進 payload.messages[].files 的 remote_ref。
pub async fn upload_file(
    http: &reqwest::Client,
    token: &str,
    filename: &str,
    bytes: &[u8],
    content_type: &str,
    parse_timeout_secs: u64,
) -> AppResult<Value> {
    let req = |method: reqwest::Method, path: &str| {
        http.request(method, format!("{BASE_URL}{path}"))
            .header("Authorization", format!("Bearer {token}"))
            .header("User-Agent", UA)
            .header("Accept", "application/json, text/plain, */*")
            .header("Referer", "https://chat.qwen.ai/")
            .header("Origin", "https://chat.qwen.ai")
            .header("Content-Type", "application/json")
    };

    // 1) getstsToken（連線偶有抖動，重試 3 次）
    let sts_body = json!({ "filename": filename, "filesize": bytes.len(), "filetype": "file" });
    let mut sts: Option<Value> = None;
    let mut last_err = String::new();
    for attempt in 0..3 {
        match req(reqwest::Method::POST, "/api/v2/files/getstsToken")
            .timeout(Duration::from_secs(20))
            .json(&sts_body)
            .send()
            .await
        {
            Ok(r) => match r.json::<Value>().await {
                Ok(v) => {
                    sts = Some(v);
                    break;
                }
                Err(e) => last_err = format!("parse failed: {e}"),
            },
            Err(e) => last_err = format!("connection failed: {e}"),
        }
        if attempt < 2 {
            tokio::time::sleep(Duration::from_millis(300 * (attempt + 1))).await;
        }
    }
    let sts = sts.ok_or_else(|| AppError::Upstream(format!("getstsToken {last_err}")))?;
    let d = sts.get("data").ok_or_else(|| AppError::Upstream("getstsToken response missing data".into()))?;
    let ak = d.get("access_key_id").and_then(|v| v.as_str()).unwrap_or("");
    let sk = d.get("access_key_secret").and_then(|v| v.as_str()).unwrap_or("");
    let stoken = d.get("security_token").and_then(|v| v.as_str()).unwrap_or("");
    let bucket = d.get("bucketname").and_then(|v| v.as_str()).unwrap_or("");
    let endpoint = d.get("endpoint").and_then(|v| v.as_str()).unwrap_or("");
    let file_path = d.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
    let file_id = d.get("file_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let region_raw = d.get("region").and_then(|v| v.as_str()).unwrap_or("");
    let region = region_raw.strip_prefix("oss-").unwrap_or(region_raw);

    if ak.is_empty() || bucket.is_empty() || file_path.is_empty() {
        return Err(AppError::Upstream("getstsToken response missing required fields".into()));
    }

    // 2) PUT 到 OSS（V4 簽名）
    let creds = StsCreds { ak, sk, token: stoken, region };
    let (authorization, xdate, content_sha) = sign_put(&creds, bucket, file_path, content_type);
    let host = format!("{bucket}.{endpoint}");
    let put_url = format!("https://{host}/{}", v4_uri_encode(file_path, false)); // URL 路徑把 '/' 編成 %2F
    let put_resp = http
        .put(&put_url)
        .header("Authorization", authorization)
        .header("x-oss-date", xdate)
        .header("x-oss-content-sha256", content_sha)
        .header("x-oss-security-token", stoken)
        .header("Content-Type", content_type)
        .timeout(Duration::from_secs(60))
        .body(bytes.to_vec())
        .send()
        .await
        .map_err(|e| AppError::Upstream(format!("OSS PUT connection failed: {e}")))?;
    if !put_resp.status().is_success() {
        let st = put_resp.status();
        let body = put_resp.text().await.unwrap_or_default();
        return Err(AppError::Upstream(format!("OSS PUT failed {st}: {}", body.chars().take(200).collect::<String>())));
    }

    // 3) parse
    let _ = req(reqwest::Method::POST, "/api/v2/files/parse")
        .timeout(Duration::from_secs(20))
        .json(&json!({ "file_id": file_id }))
        .send()
        .await;

    // 4) poll parse/status
    let deadline = std::time::Instant::now() + Duration::from_secs(parse_timeout_secs);
    let mut parse_status = "pending".to_string();
    while std::time::Instant::now() < deadline {
        let st: Value = match req(reqwest::Method::POST, "/api/v2/files/parse/status")
            .timeout(Duration::from_secs(15))
            .json(&json!({ "file_id_list": [file_id] }))
            .send()
            .await
        {
            Ok(r) => r.json().await.unwrap_or(Value::Null),
            Err(_) => Value::Null,
        };
        if let Some(s) = st.get("data").and_then(|a| a.as_array()).and_then(|a| a.first()).and_then(|x| x.get("status")).and_then(|v| v.as_str()) {
            parse_status = s.to_string();
            if s == "success" {
                break;
            }
            if s == "failed" || s == "error" {
                return Err(AppError::Upstream(format!("File parse failed status={s}")));
            }
        }
        tokio::time::sleep(Duration::from_millis(1000)).await;
    }

    // 5) Build remote_ref.
    let ms = now_millis();
    let user_id = file_path.split('/').next().unwrap_or("");
    let url = format!("https://{bucket}.{endpoint}/{file_path}");
    Ok(json!({
        "type": "file",
        "file": {
            "created_at": ms,
            "data": {},
            "filename": filename,
            "hash": Value::Null,
            "id": file_id,
            "user_id": user_id,
            "meta": { "name": filename, "size": bytes.len(), "content_type": content_type, "parse_meta": { "parse_status": parse_status } },
            "update_at": ms,
        },
        "id": file_id,
        "url": url,
        "name": filename,
        "collection_name": "",
        "progress": 0,
        "status": "uploaded",
        "greenNet": "success",
        "size": bytes.len(),
        "error": "",
        "itemId": uuid4(),
        "file_type": content_type,
        "showType": "file",
        "file_class": file_class(content_type),
        "uploadTaskId": uuid4(),
    }))
}
