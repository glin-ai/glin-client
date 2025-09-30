use reqwest::{Client, StatusCode};
use uuid::Uuid;

use crate::error::{ClientError, Result};
use super::types::*;

pub struct BackendClient {
    client: Client,
    base_url: String,
    token: Option<String>,
}

impl BackendClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            token: None,
        }
    }

    pub fn set_token(&mut self, token: &str) {
        self.token = Some(token.to_string());
    }

    /// Register a new provider
    pub async fn register(&self, request: RegisterProviderRequest) -> Result<RegisterProviderResponse> {
        let url = format!("{}/api/v1/providers/register", self.base_url);

        tracing::debug!("POST {}", url);

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            let status = response.status();
            let body = response.text().await?;
            Err(ClientError::Api(format!("Registration failed ({}): {}", status, body)))
        }
    }

    /// Send heartbeat
    pub async fn heartbeat(&self, request: ProviderHeartbeat) -> Result<()> {
        let url = format!("{}/api/v1/providers/heartbeat", self.base_url);

        tracing::debug!("POST {}", url);

        let mut req = self.client.post(&url).json(&request);

        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await?;
            Err(ClientError::Api(format!("Heartbeat failed ({}): {}", status, body)))
        }
    }

    /// Get provider details
    pub async fn get_provider(&self, id: Uuid) -> Result<Provider> {
        let url = format!("{}/api/v1/providers/{}", self.base_url, id);

        tracing::debug!("GET {}", url);

        let response = self.client
            .get(&url)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            let status = response.status();
            let body = response.text().await?;
            Err(ClientError::Api(format!("Get provider failed ({}): {}", status, body)))
        }
    }

    /// Get provider's assigned tasks
    pub async fn get_provider_tasks(&self) -> Result<Vec<ProviderTaskInfo>> {
        let url = format!("{}/api/v1/providers/tasks", self.base_url);

        tracing::debug!("GET {}", url);

        let mut req = self.client.get(&url);

        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else if response.status() == StatusCode::NOT_FOUND {
            Ok(vec![])
        } else {
            let status = response.status();
            let body = response.text().await?;
            Err(ClientError::Api(format!("Get tasks failed ({}): {}", status, body)))
        }
    }

    /// Submit gradient CID after training completion
    pub async fn submit_gradient(&self, request: super::types::SubmitGradientRequest) -> Result<()> {
        let url = format!("{}/api/v1/gradients/submit", self.base_url);

        tracing::debug!("POST {}", url);

        let mut req = self.client.post(&url).json(&request);

        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await?;

        if response.status().is_success() {
            tracing::info!("Gradient submitted successfully for task {}", request.task_id);
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await?;
            Err(ClientError::Api(format!("Submit gradient failed ({}): {}", status, body)))
        }
    }
}
