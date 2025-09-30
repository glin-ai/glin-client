use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

use crate::error::{ClientError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub provider: ProviderConfig,
    pub backend: BackendConfig,
    pub worker: WorkerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: Option<Uuid>,
    pub name: String,
    pub wallet_address: String,
    pub api_key: Option<String>,
    pub jwt_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub heartbeat_interval_secs: u64,
    pub task_poll_interval_secs: u64,
    pub max_concurrent_tasks: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: ProviderConfig {
                id: None,
                name: String::new(),
                wallet_address: String::new(),
                api_key: None,
                jwt_token: None,
            },
            backend: BackendConfig {
                url: "http://localhost:3000".to_string(),
            },
            worker: WorkerConfig {
                heartbeat_interval_secs: 30,
                task_poll_interval_secs: 10,
                max_concurrent_tasks: 1,
            },
        }
    }
}

impl Config {
    /// Get the config file path
    pub fn config_path() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("ai", "glin", "client")
            .ok_or_else(|| ClientError::Config("Could not determine config directory".to_string()))?;

        Ok(proj_dirs.config_dir().join("config.toml"))
    }

    /// Get the data directory path
    pub fn data_dir() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("ai", "glin", "client")
            .ok_or_else(|| ClientError::Config("Could not determine data directory".to_string()))?;

        Ok(proj_dirs.data_dir().to_path_buf())
    }

    /// Load configuration from file
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            return Err(ClientError::NotRegistered);
        }

        let contents = std::fs::read_to_string(&config_path)
            .map_err(|e| ClientError::Config(format!("Failed to read config file: {}", e)))?;

        let config: Config = toml::from_str(&contents)?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)
            .map_err(|e| ClientError::Config(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(&config_path, contents)?;

        tracing::info!("Configuration saved to: {}", config_path.display());

        Ok(())
    }

    /// Check if provider is registered
    pub fn is_registered(&self) -> bool {
        self.provider.id.is_some() && self.provider.api_key.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.backend.url, "http://localhost:3000");
        assert_eq!(config.worker.heartbeat_interval_secs, 30);
        assert!(!config.is_registered());
    }
}
