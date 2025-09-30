use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Hardware detection error: {0}")]
    Hardware(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("GPU detection error: {0}")]
    GpuDetection(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JWT error: {0}")]
    Jwt(#[from] jsonwebtoken::errors::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Not registered: Please run 'glin-client register' first")]
    NotRegistered,

    #[error("Already registered: Provider ID: {0}")]
    AlreadyRegistered(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;
