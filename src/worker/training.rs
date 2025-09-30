use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use uuid::Uuid;

use crate::error::{ClientError, Result};
use crate::storage::{CacheManager, IpfsClient};

pub struct TrainingExecutor {
    ipfs: IpfsClient,
    cache: CacheManager,
    python_path: String,
}

#[derive(Debug, Clone)]
pub struct TrainingTask {
    pub task_id: Uuid,
    pub model_cid: String,
    pub dataset_url: String,
    pub config: TrainingConfig,
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
}

#[derive(Debug)]
pub struct TrainingResult {
    pub gradient_cid: String,
    pub metrics: TrainingMetrics,
    pub logs: Vec<String>,
}

#[derive(Debug)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub duration_secs: u64,
}

impl TrainingExecutor {
    pub fn new(cache_dir: PathBuf, ipfs_gateway: Option<String>) -> Result<Self> {
        let python_path = std::env::var("PYTHON_PATH").unwrap_or_else(|_| "python3".to_string());

        Ok(Self {
            ipfs: IpfsClient::new(ipfs_gateway),
            cache: CacheManager::new(cache_dir),
            python_path,
        })
    }

    /// Execute a training task
    pub async fn execute(&self, task: &TrainingTask) -> Result<TrainingResult> {
        tracing::info!("Starting training for task {}", task.task_id);

        // Initialize cache directories
        self.cache.init().await?;

        // 1. Download model from IPFS
        let model_path = self.download_model(&task.model_cid).await?;

        // 2. Download dataset
        let dataset_path = self.download_dataset(&task.dataset_url).await?;

        // 3. Prepare output directory
        let output_dir = self.cache.output_path(&task.task_id.to_string());
        fs::create_dir_all(&output_dir).await?;

        // 4. Run training script
        let (logs, metrics) = self.run_training_script(
            &model_path,
            &dataset_path,
            &output_dir,
            &task.config,
        ).await?;

        // 5. Upload gradients to IPFS
        let gradient_path = output_dir.join("gradients.pt");
        let gradient_cid = self.ipfs.upload(&gradient_path).await?;

        tracing::info!("Training completed for task {}, gradient CID: {}", task.task_id, gradient_cid);

        Ok(TrainingResult {
            gradient_cid,
            metrics,
            logs,
        })
    }

    /// Download model from IPFS (with caching)
    async fn download_model(&self, cid: &str) -> Result<PathBuf> {
        let cache_path = self.cache.model_path(cid);

        // Check cache first
        if self.cache.has_model(cid).await {
            tracing::info!("Using cached model: {}", cid);
            return Ok(cache_path);
        }

        // Download from IPFS
        tracing::info!("Downloading model from IPFS: {}", cid);
        self.ipfs.download(cid, &cache_path).await
    }

    /// Download dataset (supports both IPFS and HTTP)
    async fn download_dataset(&self, url: &str) -> Result<PathBuf> {
        if url.starts_with("ipfs://") {
            let cid = url.strip_prefix("ipfs://").unwrap();
            let cache_path = self.cache.dataset_path(cid);

            if self.cache.has_dataset(cid).await {
                tracing::info!("Using cached dataset: {}", cid);
                return Ok(cache_path);
            }

            tracing::info!("Downloading dataset from IPFS: {}", cid);
            self.ipfs.download(cid, &cache_path).await
        } else {
            // HTTP/HTTPS download
            let filename = url.split('/').last().unwrap_or("dataset.zip");
            let cache_path = self.cache.dataset_path(filename);

            if cache_path.exists() {
                tracing::info!("Using cached dataset: {}", filename);
                return Ok(cache_path);
            }

            tracing::info!("Downloading dataset from URL: {}", url);
            self.download_http(url, &cache_path).await?;
            Ok(cache_path)
        }
    }

    async fn download_http(&self, url: &str, output_path: &Path) -> Result<()> {
        let response = reqwest::get(url)
            .await
            .map_err(|e| ClientError::Storage(format!("HTTP download failed: {}", e)))?;

        let bytes = response.bytes()
            .await
            .map_err(|e| ClientError::Storage(format!("Failed to read response: {}", e)))?;

        fs::write(output_path, bytes).await?;
        Ok(())
    }

    /// Run the Python training script
    async fn run_training_script(
        &self,
        model_path: &Path,
        dataset_path: &Path,
        output_dir: &Path,
        config: &TrainingConfig,
    ) -> Result<(Vec<String>, TrainingMetrics)> {
        tracing::info!("Running Python training script");

        // Get the Python script path (should be in python/ directory)
        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python")
            .join("train.py");

        if !script_path.exists() {
            return Err(ClientError::Training(
                "Training script not found. Please ensure python/train.py exists.".to_string()
            ));
        }

        let start = std::time::Instant::now();

        let mut child = Command::new(&self.python_path)
            .arg(&script_path)
            .arg("--model").arg(model_path)
            .arg("--dataset").arg(dataset_path)
            .arg("--output").arg(output_dir)
            .arg("--epochs").arg(config.epochs.to_string())
            .arg("--batch-size").arg(config.batch_size.to_string())
            .arg("--learning-rate").arg(config.learning_rate.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ClientError::Training(format!("Failed to spawn Python process: {}", e)))?;

        let stdout = child.stdout.take()
            .ok_or_else(|| ClientError::Training("Failed to capture stdout".to_string()))?;
        let stderr = child.stderr.take()
            .ok_or_else(|| ClientError::Training("Failed to capture stderr".to_string()))?;

        // Capture logs asynchronously
        let mut logs = Vec::new();
        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        loop {
            tokio::select! {
                line = stdout_reader.next_line() => {
                    match line {
                        Ok(Some(line)) => {
                            tracing::info!("[TRAINING] {}", line);
                            logs.push(line);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            tracing::error!("Error reading stdout: {}", e);
                            break;
                        }
                    }
                }
                line = stderr_reader.next_line() => {
                    match line {
                        Ok(Some(line)) => {
                            tracing::warn!("[TRAINING ERROR] {}", line);
                            logs.push(format!("ERROR: {}", line));
                        }
                        Ok(None) => {},
                        Err(e) => {
                            tracing::error!("Error reading stderr: {}", e);
                        }
                    }
                }
            }
        }

        // Wait for process to complete
        let status = child.wait().await
            .map_err(|e| ClientError::Training(format!("Failed to wait for process: {}", e)))?;

        if !status.success() {
            return Err(ClientError::Training(format!(
                "Training script failed with exit code: {:?}", status.code()
            )));
        }

        let duration = start.elapsed().as_secs();

        // Parse metrics from output file
        let metrics = self.parse_metrics(output_dir).await?;

        Ok((logs, TrainingMetrics {
            loss: metrics.loss,
            accuracy: metrics.accuracy,
            duration_secs: duration,
        }))
    }

    async fn parse_metrics(&self, output_dir: &Path) -> Result<TrainingMetrics> {
        let metrics_path = output_dir.join("metrics.json");

        if !metrics_path.exists() {
            // Return default metrics if file doesn't exist
            return Ok(TrainingMetrics {
                loss: 0.0,
                accuracy: 0.0,
                duration_secs: 0,
            });
        }

        let content = fs::read_to_string(&metrics_path).await?;
        let metrics: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| ClientError::Training(format!("Failed to parse metrics: {}", e)))?;

        Ok(TrainingMetrics {
            loss: metrics["loss"].as_f64().unwrap_or(0.0),
            accuracy: metrics["accuracy"].as_f64().unwrap_or(0.0),
            duration_secs: 0,
        })
    }
}
