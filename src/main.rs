//! qwen2api-rs entrypoint: axum app assembly, lifecycle, and routes.
// Some fields/methods are kept for future upstream sync work, so dead_code is relaxed.
#![allow(dead_code)]

mod account;
mod api;
mod auth;
mod config;
mod context;
mod db;
mod error;
mod execution;
mod media;
mod request;
mod state;
mod stats;
mod toolcall;
mod upstream;
mod util;

use axum::routing::{delete, get, post};
use axum::Router;
use config::Settings;
use state::AppStateInner;
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    let settings = Settings::from_env();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&settings.log_level)),
        )
        .init();

    tracing::info!("Starting qwen2api-rs gateway ...");
    let port = settings.port;

    let state = AppStateInner::new(settings).await;

    // Start the chat_id prewarm pool.
    state.chat_id_pool.start();

    // Start the media queue worker for image/video generation and local backups.
    state.media_queue.clone().start(state.clone());

    // Optional connection keepalive. Disabled by default because upstream risk controls can be sensitive.
    if state.settings.conn_keepalive_seconds > 0 {
        let state3 = state.clone();
        let interval = state.settings.conn_keepalive_seconds;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
                if let Some(token) = state3.pool.any_valid_token().await {
                    let _ = state3.client.verify_token(&token).await;
                }
            }
        });
        tracing::info!("Connection keepalive enabled: ping upstream every {interval}s");
    }

    // Token refresh worker: refresh chat.qwen.ai JWTs before they expire.
    if state.settings.token_refresh_interval_hours > 0 {
        let state_w = state.clone();
        let interval_secs = state.settings.token_refresh_interval_hours * 3600;
        let ahead_days = state.settings.token_refresh_ahead_days;
        let batch = state.settings.token_refresh_batch_per_cycle;
        let jmin = state.settings.token_refresh_jitter_min_ms;
        let jmax = state.settings.token_refresh_jitter_max_ms.max(jmin);
        tokio::spawn(async move {
            // Delay startup work so cold start stays light.
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            loop {
                let cycle_start = std::time::Instant::now();
                let now = crate::util::now_unix();
                let cutoff = now + ahead_days * 86400;
                let accounts = state_w.pool.list().await;
                let total = accounts.len();
                // Select accounts with a password and a JWT expiring before the cutoff.
                let mut due: Vec<(String, String)> = accounts
                    .into_iter()
                    .filter(|a| !a.password.is_empty() && !a.token.is_empty())
                    .filter_map(|a| crate::util::jwt_exp(&a.token).map(|exp| (a, exp)))
                    .filter(|(_, exp)| *exp <= cutoff)
                    .map(|(a, _)| (a.email, a.password))
                    .collect();
                let due_total = due.len();
                if due.len() > batch {
                    due.truncate(batch);
                }
                tracing::info!(
                    "[refresh-worker] scanned {total} accounts; {due_total} expire within {ahead_days} days; processing {} this round",
                    due.len()
                );
                let mut ok_n = 0usize;
                let mut fail_n = 0usize;
                for (email, password) in due {
                    match state_w.client.signin(&email, &password).await {
                        Ok(new_token) => {
                            let _ = state_w.pool.replace_token(&email, new_token).await;
                            ok_n += 1;
                        }
                        Err(e) => {
                            fail_n += 1;
                            let msg = e.to_string();
                            state_w.pool.apply_verify(&email, false, "auth_error", &msg).await;
                            tracing::warn!("[refresh-worker] {email} refresh failed: {msg}");
                        }
                    }
                    let span = jmax.saturating_sub(jmin).max(1);
                    let jitter = jmin + (rand::random::<u64>() % span);
                    tokio::time::sleep(std::time::Duration::from_millis(jitter)).await;
                }
                tracing::info!(
                    "[refresh-worker] round finished ok={ok_n} fail={fail_n} elapsed={:?}; sleeping {}h",
                    cycle_start.elapsed(),
                    interval_secs / 3600
                );
                tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            }
        });
        tracing::info!(
            "Token refresh worker enabled: interval={}h ahead={ahead_days}d batch={batch} jitter={jmin}-{jmax}ms",
            state.settings.token_refresh_interval_hours
        );
    }

    // Best-effort upstream model cache warmup.
    {
        let state2 = state.clone();
        tokio::spawn(async move {
            if let Some(token) = state2.pool.any_valid_token().await {
                let models = state2.client.list_models(&token).await;
                if let Some(first) = models.first().and_then(|m| m.get("id")).and_then(|v| v.as_str()) {
                    tracing::info!("Sample upstream model: {first}");
                }
                let mut cache = state2.upstream_models.write().await;
                cache.data = models;
                cache.fetched_at = crate::util::now_secs();
            }
        });
    }

    // Generated media is served from /media/{file}.
    let media_service = ServeDir::new(state.settings.media_dir.clone());

    let app = Router::new()
        // OpenAI Chat Completions
        .route("/v1/chat/completions", post(api::openai::chat_completions))
        .route("/chat/completions", post(api::openai::chat_completions))
        // OpenAI Responses
        .route("/v1/responses", post(api::responses::create))
        .route("/responses", post(api::responses::create))
        // Anthropic Messages
        .route("/v1/messages", post(api::anthropic::messages))
        .route("/messages", post(api::anthropic::messages))
        .route("/anthropic/v1/messages", post(api::anthropic::messages))
        .route("/v1/messages/count_tokens", post(api::anthropic::count_tokens))
        .route("/messages/count_tokens", post(api::anthropic::count_tokens))
        .route("/anthropic/v1/messages/count_tokens", post(api::anthropic::count_tokens))
        // Gemini paths include {model}:{action}.
        .route("/v1beta/models/{model_id}", post(api::gemini::generate))
        .route("/models/{model_id}", post(api::gemini::generate))
        .route("/v1/models/{model_id}", post(api::gemini::generate))
        // OpenAI Images / Embeddings
        .route("/v1/images/generations", post(api::images::generate))
        .route("/images/generations", post(api::images::generate))
        .route("/v1/videos/generations", post(api::videos::generate))
        .route("/videos/generations", post(api::videos::generate))
        .route("/v1/embeddings", post(api::embeddings::create))
        .route("/embeddings", post(api::embeddings::create))
        // Files
        .route("/v1/files", post(api::files::upload))
        .route("/api/files/upload", post(api::files::upload))
        .route("/v1/files/{file_id}", delete(api::files::delete))
        .route("/api/files/{file_id}", delete(api::files::delete))
        // Models
        .route("/v1/models", get(api::models::list_models))
        .route("/v1/models/{model_id}", get(api::models::get_model))
        // Probes and JSON root.
        .route("/", get(api::probes::root))
        .route("/healthz", get(api::probes::healthz))
        .route("/readyz", get(api::probes::readyz))
        .route("/api", get(api::probes::root))
        .nest_service("/media", media_service)
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("failed to bind port");
    tracing::info!("qwen2api-rs is listening on http://0.0.0.0:{port}");
    axum::serve(listener, app).await.expect("server error");
}
