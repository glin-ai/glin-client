use std::path::{Path, PathBuf};
use tokio::fs;

use crate::error::Result;

pub struct CacheManager {
    cache_dir: PathBuf,
}

impl CacheManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get path for a cached model
    pub fn model_path(&self, cid: &str) -> PathBuf {
        self.cache_dir.join("models").join(cid)
    }

    /// Get path for a cached dataset
    pub fn dataset_path(&self, cid: &str) -> PathBuf {
        self.cache_dir.join("datasets").join(cid)
    }

    /// Get path for training outputs
    pub fn output_path(&self, task_id: &str) -> PathBuf {
        self.cache_dir.join("outputs").join(task_id)
    }

    /// Initialize cache directories
    pub async fn init(&self) -> Result<()> {
        fs::create_dir_all(self.cache_dir.join("models")).await?;
        fs::create_dir_all(self.cache_dir.join("datasets")).await?;
        fs::create_dir_all(self.cache_dir.join("outputs")).await?;
        Ok(())
    }

    /// Check if a model is cached
    pub async fn has_model(&self, cid: &str) -> bool {
        self.model_path(cid).exists()
    }

    /// Check if a dataset is cached
    pub async fn has_dataset(&self, cid: &str) -> bool {
        self.dataset_path(cid).exists()
    }

    /// Clean up old cache entries
    pub async fn cleanup(&self, max_age_days: u64) -> Result<()> {
        let max_age = std::time::Duration::from_secs(max_age_days * 24 * 60 * 60);
        let now = std::time::SystemTime::now();

        for dir in &["models", "datasets", "outputs"] {
            let dir_path = self.cache_dir.join(dir);
            if !dir_path.exists() {
                continue;
            }

            let mut entries = fs::read_dir(&dir_path).await?;
            while let Some(entry) = entries.next_entry().await? {
                let metadata = entry.metadata().await?;
                if let Ok(modified) = metadata.modified() {
                    if let Ok(age) = now.duration_since(modified) {
                        if age > max_age {
                            let path = entry.path();
                            tracing::info!("Cleaning up old cache entry: {:?}", path);
                            if path.is_dir() {
                                fs::remove_dir_all(&path).await?;
                            } else {
                                fs::remove_file(&path).await?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get cache size in bytes
    pub async fn cache_size(&self) -> Result<u64> {
        let mut total_size = 0u64;

        for dir in &["models", "datasets", "outputs"] {
            let dir_path = self.cache_dir.join(dir);
            if !dir_path.exists() {
                continue;
            }

            total_size += self.dir_size(&dir_path).await?;
        }

        Ok(total_size)
    }

    fn dir_size<'a>(&'a self, path: &'a Path) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64>> + 'a>> {
        Box::pin(async move {
            let mut size = 0u64;
            let mut entries = fs::read_dir(path).await?;

            while let Some(entry) = entries.next_entry().await? {
                let metadata = entry.metadata().await?;
                if metadata.is_dir() {
                    size += self.dir_size(&entry.path()).await?;
                } else {
                    size += metadata.len();
                }
            }

            Ok(size)
        })
    }
}
