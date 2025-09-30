mod training;
mod gpu_monitor;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tokio::signal;
use uuid::Uuid;

use crate::{
    api::BackendClient,
    config::Config,
    error::{ClientError, Result},
};

pub use training::{TrainingExecutor, TrainingTask, TrainingConfig};
pub use gpu_monitor::GpuMonitor;

pub struct Worker {
    config: Config,
    client: BackendClient,
    active_tasks: Arc<RwLock<Vec<Uuid>>>,
    shutdown: Arc<RwLock<bool>>,
    gpu_monitor: GpuMonitor,
}

impl Worker {
    pub async fn new(config: Config) -> Result<Self> {
        let mut client = BackendClient::new(&config.backend.url);

        if let Some(token) = &config.provider.jwt_token {
            client.set_token(token);
        }

        Ok(Self {
            config,
            client,
            active_tasks: Arc::new(RwLock::new(Vec::new())),
            shutdown: Arc::new(RwLock::new(false)),
            gpu_monitor: GpuMonitor::new(),
        })
    }

    pub async fn run(&self) -> Result<()> {
        let provider_id = self.config.provider.id
            .ok_or_else(|| ClientError::NotRegistered)?;

        tracing::info!("Worker started for provider {}", provider_id);

        // Setup graceful shutdown handler
        let shutdown_flag = Arc::clone(&self.shutdown);
        tokio::spawn(async move {
            match signal::ctrl_c().await {
                Ok(()) => {
                    tracing::info!("Received shutdown signal (Ctrl+C)");
                    *shutdown_flag.write().await = true;
                }
                Err(err) => {
                    tracing::error!("Unable to listen for shutdown signal: {}", err);
                }
            }
        });

        // Main worker loop
        loop {
            // Check shutdown flag
            if *self.shutdown.read().await {
                tracing::info!("Shutting down worker...");
                self.graceful_shutdown().await?;
                break;
            }

            // Send heartbeat
            if let Err(e) = self.send_heartbeat(provider_id).await {
                tracing::error!("Failed to send heartbeat: {}", e);
            }

            // Poll for tasks
            match self.client.get_provider_tasks().await {
                Ok(tasks) => {
                    if !tasks.is_empty() {
                        tracing::info!("Found {} assigned tasks", tasks.len());

                        // Execute tasks concurrently
                        let mut handles = Vec::new();

                        for task in tasks {
                            // Check if task is already running
                            let active = self.active_tasks.read().await;
                            if active.contains(&task.id) {
                                tracing::debug!("Task {} already running, skipping", task.id);
                                continue;
                            }
                            drop(active);

                            // Mark task as active
                            self.active_tasks.write().await.push(task.id);

                            // Spawn task execution
                            let task_id = task.id;
                            let task_name = task.name.clone();
                            let active_tasks = Arc::clone(&self.active_tasks);
                            let shutdown_flag = Arc::clone(&self.shutdown);

                            let handle = tokio::spawn(async move {
                                tracing::info!("Processing task: {} ({})", task_name, task_id);

                                // Execute training
                                if let Err(e) = Self::execute_task(&task_id, shutdown_flag).await {
                                    tracing::error!("Task {} failed: {}", task_id, e);
                                }

                                // Remove from active tasks
                                active_tasks.write().await.retain(|id| id != &task_id);
                                tracing::info!("Task {} completed", task_id);
                            });

                            handles.push(handle);
                        }

                        // Don't wait for tasks to complete, they run in background
                        // The active_tasks list keeps track of them
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to fetch tasks: {}", e);
                }
            }

            // Wait before next poll
            sleep(Duration::from_secs(self.config.worker.heartbeat_interval_secs)).await;
        }

        Ok(())
    }

    async fn send_heartbeat(&self, provider_id: Uuid) -> Result<()> {
        // Get current active task IDs
        let current_tasks = self.active_tasks.read().await.clone();

        let heartbeat = crate::api::types::ProviderHeartbeat {
            provider_id,
            current_tasks,
            cpu_usage: self.get_cpu_usage(),
            gpu_usage: self.get_gpu_usage(),
            memory_usage: self.get_memory_usage(),
            temperature: self.get_gpu_temperature(),
            available_vram_gb: self.get_available_vram(),
        };

        self.client.heartbeat(heartbeat).await?;
        tracing::debug!("Heartbeat sent successfully");
        Ok(())
    }

    /// Execute a single task (static method for spawning)
    async fn execute_task(task_id: &Uuid, shutdown_flag: Arc<RwLock<bool>>) -> Result<()> {
        // Check if shutdown requested before starting
        if *shutdown_flag.read().await {
            tracing::warn!("Task {} skipped due to shutdown", task_id);
            return Ok(());
        }

        // For MVP, we use dummy training with simulated work
        // TODO: In v0.2.0, integrate real TrainingExecutor here
        // Example:
        // let executor = TrainingExecutor::new(cache_dir, ipfs_gateway)?;
        // let task = TrainingTask {
        //     task_id: *task_id,
        //     model_cid: "QmXXX...".to_string(),
        //     dataset_url: "ipfs://QmYYY...".to_string(),
        //     config: TrainingConfig {
        //         epochs: 1,
        //         batch_size: 32,
        //         learning_rate: 0.001,
        //     },
        // };
        // let result = executor.execute(&task).await?;

        tracing::info!("Executing training for task {}", task_id);

        // Simulate training work in chunks, checking shutdown flag
        for i in 0..10 {
            // Check if shutdown requested
            if *shutdown_flag.read().await {
                tracing::warn!("Task {} interrupted by shutdown", task_id);
                return Ok(());
            }

            sleep(Duration::from_millis(500)).await;
            tracing::debug!("Task {} progress: {}%", task_id, (i + 1) * 10);
        }

        tracing::info!("Training completed for task {}", task_id);

        // TODO: Submit gradient CID to backend
        // Example:
        // let gradient_request = SubmitGradientRequest {
        //     task_id: *task_id,
        //     provider_id,
        //     gradient_cid: result.gradient_cid,
        //     metrics: GradientMetrics {
        //         loss: result.metrics.loss,
        //         accuracy: result.metrics.accuracy,
        //         training_duration_secs: result.metrics.duration_secs,
        //         compression_method: "quantize".to_string(),
        //     },
        // };
        // client.submit_gradient(gradient_request).await?;

        Ok(())
    }

    /// Gracefully shutdown worker, waiting for active tasks
    async fn graceful_shutdown(&self) -> Result<()> {
        println!("\n🛑 Graceful shutdown initiated...");

        // Wait for all active tasks to complete (with timeout)
        let max_wait = Duration::from_secs(30);
        let start = std::time::Instant::now();

        loop {
            let active = self.active_tasks.read().await;
            let count = active.len();

            if count == 0 {
                println!("✅ All tasks completed");
                break;
            }

            if start.elapsed() > max_wait {
                println!("⚠️  Timeout waiting for {} tasks to complete", count);
                println!("   Force shutting down...");
                break;
            }

            println!("   Waiting for {} active task(s) to complete...", count);
            drop(active);
            sleep(Duration::from_secs(2)).await;
        }

        println!("👋 Worker shutdown complete");
        Ok(())
    }

    fn get_cpu_usage(&self) -> f32 {
        // TODO: Implement actual CPU usage monitoring with sysinfo
        0.0
    }

    fn get_gpu_usage(&self) -> f32 {
        self.gpu_monitor.get_usage()
    }

    fn get_memory_usage(&self) -> f32 {
        self.gpu_monitor.get_memory_usage()
    }

    fn get_gpu_temperature(&self) -> f32 {
        self.gpu_monitor.get_temperature()
    }

    fn get_available_vram(&self) -> f32 {
        self.gpu_monitor.get_available_vram()
    }
}
