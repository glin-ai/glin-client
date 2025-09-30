use std::process::Command;
use crate::error::{ClientError, Result};

#[derive(Debug, Clone)]
pub struct GpuStats {
    pub usage_percent: f32,
    pub memory_used_mb: u32,
    pub memory_total_mb: u32,
    pub temperature_c: f32,
    pub power_draw_w: f32,
}

pub struct GpuMonitor;

impl GpuMonitor {
    pub fn new() -> Self {
        Self
    }

    /// Get current GPU statistics
    pub fn get_stats(&self) -> Result<GpuStats> {
        // Try nvidia-smi first
        if let Ok(stats) = self.get_nvidia_stats() {
            return Ok(stats);
        }

        // Fallback to default values if no GPU monitoring available
        Ok(GpuStats {
            usage_percent: 0.0,
            memory_used_mb: 0,
            memory_total_mb: 0,
            temperature_c: 0.0,
            power_draw_w: 0.0,
        })
    }

    fn get_nvidia_stats(&self) -> Result<GpuStats> {
        let output = Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw")
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|e| ClientError::GpuDetection(format!("Failed to run nvidia-smi: {}", e)))?;

        if !output.status.success() {
            return Err(ClientError::GpuDetection("nvidia-smi failed".to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().next()
            .ok_or_else(|| ClientError::GpuDetection("No GPU stats available".to_string()))?;

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            return Err(ClientError::GpuDetection("Unexpected nvidia-smi output".to_string()));
        }

        Ok(GpuStats {
            usage_percent: parts[0].parse().unwrap_or(0.0),
            memory_used_mb: parts[1].parse().unwrap_or(0),
            memory_total_mb: parts[2].parse().unwrap_or(0),
            temperature_c: parts[3].parse().unwrap_or(0.0),
            power_draw_w: parts[4].parse().unwrap_or(0.0),
        })
    }

    /// Get GPU usage percentage (0-100)
    pub fn get_usage(&self) -> f32 {
        self.get_stats()
            .map(|s| s.usage_percent)
            .unwrap_or(0.0)
    }

    /// Get available VRAM in GB
    pub fn get_available_vram(&self) -> f32 {
        self.get_stats()
            .map(|s| {
                let available_mb = s.memory_total_mb.saturating_sub(s.memory_used_mb);
                available_mb as f32 / 1024.0
            })
            .unwrap_or(0.0)
    }

    /// Get GPU temperature in Celsius
    pub fn get_temperature(&self) -> f32 {
        self.get_stats()
            .map(|s| s.temperature_c)
            .unwrap_or(0.0)
    }

    /// Get memory usage percentage
    pub fn get_memory_usage(&self) -> f32 {
        self.get_stats()
            .map(|s| {
                if s.memory_total_mb > 0 {
                    (s.memory_used_mb as f32 / s.memory_total_mb as f32) * 100.0
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_monitor_creation() {
        let monitor = GpuMonitor::new();
        let stats = monitor.get_stats();
        assert!(stats.is_ok());
    }
}
