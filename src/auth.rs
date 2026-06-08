//! API key authentication for the compatibility endpoints.

use crate::error::{AppError, AppResult};
use crate::state::AppState;
use axum::http::HeaderMap;
use std::collections::HashMap;

/// Auth context resolved from the request.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub token: String,
}

/// Extract API key from Authorization, x-api-key, x-goog-api-key, or query.
pub fn extract_api_token(headers: &HeaderMap, query: &HashMap<String, String>) -> Option<String> {
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(rest) = auth.strip_prefix("Bearer ") {
            let t = rest.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        } else if !auth.trim().is_empty() {
            return Some(auth.trim().to_string());
        }
    }
    if let Some(k) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    if let Some(k) = headers.get("x-goog-api-key").and_then(|v| v.to_str().ok()) {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    if let Some(k) = query.get("key").or_else(|| query.get("api_key")) {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    None
}

/// Resolve and validate the caller API key.
pub async fn resolve_auth(
    state: &AppState,
    headers: &HeaderMap,
    query: &HashMap<String, String>,
) -> AppResult<AuthContext> {
    let token = extract_api_token(headers, query)
        .ok_or_else(|| AppError::Unauthorized("Missing API key".into()))?;

    if token != state.settings.api_key {
        return Err(AppError::Unauthorized("Invalid API key".into()));
    }

    Ok(AuthContext { token })
}

/// Compatibility hook for API handlers that report usage after completion.
pub async fn add_used_tokens(_state: &AppState, _token: &str, _delta: i64) {
    // Quotas/users were removed with the admin API. Keep this as a no-op so
    // protocol handlers can stay focused on response translation.
}
