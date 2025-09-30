use std::path::{Path, PathBuf};
use reqwest::Client;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::error::{ClientError, Result};

pub struct IpfsClient {
    client: Client,
    gateway_url: String,
}

impl IpfsClient {
    pub fn new(gateway_url: Option<String>) -> Self {
        let gateway_url = gateway_url.unwrap_or_else(|| "https://ipfs.io".to_string());

        Self {
            client: Client::new(),
            gateway_url,
        }
    }

    /// Download a file from IPFS by CID
    pub async fn download(&self, cid: &str, output_path: &Path) -> Result<PathBuf> {
        let url = format!("{}/ipfs/{}", self.gateway_url, cid);

        tracing::info!("Downloading from IPFS: {} -> {:?}", cid, output_path);

        // Create parent directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Download with retry logic
        let mut attempts = 0;
        let max_attempts = 3;

        loop {
            attempts += 1;

            match self.try_download(&url, output_path).await {
                Ok(path) => {
                    tracing::info!("Successfully downloaded {} ({} bytes)", cid,
                        fs::metadata(&path).await?.len());
                    return Ok(path);
                }
                Err(e) if attempts < max_attempts => {
                    tracing::warn!("Download attempt {} failed: {}, retrying...", attempts, e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
                Err(e) => {
                    return Err(ClientError::Storage(format!(
                        "Failed to download {} after {} attempts: {}", cid, attempts, e
                    )));
                }
            }
        }
    }

    async fn try_download(&self, url: &str, output_path: &Path) -> Result<PathBuf> {
        let response = self.client
            .get(url)
            .send()
            .await
            .map_err(|e| ClientError::Storage(format!("IPFS request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ClientError::Storage(format!(
                "IPFS gateway returned status: {}", response.status()
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| ClientError::Storage(format!("Failed to read response: {}", e)))?;

        let mut file = fs::File::create(output_path).await?;
        file.write_all(&bytes).await?;

        Ok(output_path.to_path_buf())
    }

    /// Upload a file to IPFS (requires local IPFS node)
    pub async fn upload(&self, file_path: &Path) -> Result<String> {
        tracing::info!("Uploading to IPFS: {:?}", file_path);

        // Read file
        let file_data = fs::read(file_path).await?;

        // Upload to IPFS API (localhost:5001)
        let ipfs_api_url = "http://localhost:5001/api/v0/add";

        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(file_data)
                .file_name(file_path.file_name().unwrap().to_string_lossy().to_string()));

        let response = self.client
            .post(ipfs_api_url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| ClientError::Storage(format!("IPFS upload failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ClientError::Storage(format!(
                "IPFS upload returned status: {}", response.status()
            )));
        }

        let result: serde_json::Value = response.json().await
            .map_err(|e| ClientError::Storage(format!("Failed to parse IPFS response: {}", e)))?;

        let cid = result["Hash"]
            .as_str()
            .ok_or_else(|| ClientError::Storage("No Hash in IPFS response".to_string()))?
            .to_string();

        tracing::info!("Successfully uploaded to IPFS: {}", cid);
        Ok(cid)
    }

    /// Check if IPFS content is accessible
    pub async fn is_accessible(&self, cid: &str) -> bool {
        let url = format!("{}/ipfs/{}", self.gateway_url, cid);

        match self.client.head(&url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipfs_client_creation() {
        let client = IpfsClient::new(None);
        assert!(client.gateway_url.contains("ipfs.io"));
    }
}
