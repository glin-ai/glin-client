use std::process::Command;
use sysinfo::System;

use crate::api::types::HardwareInfo;
use crate::error::{ClientError, Result};

pub struct HardwareDetector {
    sys: System,
}

impl HardwareDetector {
    pub fn new() -> Result<Self> {
        let sys = System::new_all();
        Ok(Self { sys })
    }

    pub fn detect(&self) -> Result<HardwareInfo> {
        let gpu_info = self.detect_gpu()?;
        let cpu_info = self.detect_cpu();
        let os_info = self.detect_os();

        Ok(HardwareInfo {
            gpu_model: gpu_info.model,
            gpu_count: gpu_info.count,
            vram_gb: gpu_info.vram_gb,
            compute_capability: gpu_info.compute_capability,
            cpu_model: cpu_info.model,
            cpu_cores: cpu_info.cores,
            ram_gb: cpu_info.ram_gb,
            bandwidth_mbps: self.estimate_bandwidth(),
            os: os_info.name,
            driver_version: gpu_info.driver_version,
            cuda_version: gpu_info.cuda_version,
        })
    }

    fn detect_gpu(&self) -> Result<GpuInfo> {
        // Try nvidia-smi first (most common for ML workloads)
        if let Ok(nvidia_info) = self.detect_nvidia_gpu() {
            return Ok(nvidia_info);
        }

        // Fall back to basic detection
        tracing::warn!("Could not detect GPU via nvidia-smi, using fallback detection");
        Ok(GpuInfo {
            model: "Unknown GPU".to_string(),
            count: 0,
            vram_gb: 0,
            compute_capability: 0.0,
            driver_version: "Unknown".to_string(),
            cuda_version: None,
        })
    }

    fn detect_nvidia_gpu(&self) -> Result<GpuInfo> {
        // Run nvidia-smi to get GPU info
        let output = Command::new("nvidia-smi")
            .arg("--query-gpu=name,memory.total,count,driver_version,compute_cap")
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|e| ClientError::Hardware(format!("Failed to run nvidia-smi: {}", e)))?;

        if !output.status.success() {
            return Err(ClientError::Hardware("nvidia-smi failed".to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().next()
            .ok_or_else(|| ClientError::Hardware("No GPU detected".to_string()))?;

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            return Err(ClientError::Hardware("Unexpected nvidia-smi output format".to_string()));
        }

        let model = parts[0].to_string();
        let vram_mb: i32 = parts[1].parse()
            .map_err(|_| ClientError::Hardware("Failed to parse VRAM".to_string()))?;
        let count: i32 = parts[2].parse()
            .map_err(|_| ClientError::Hardware("Failed to parse GPU count".to_string()))?;
        let driver_version = parts[3].to_string();
        let compute_capability: f32 = parts[4].parse()
            .map_err(|_| ClientError::Hardware("Failed to parse compute capability".to_string()))?;

        // Try to get CUDA version
        let cuda_version = self.detect_cuda_version().ok();

        Ok(GpuInfo {
            model,
            count,
            vram_gb: vram_mb / 1024,
            compute_capability,
            driver_version,
            cuda_version,
        })
    }

    fn detect_cuda_version(&self) -> Result<String> {
        let output = Command::new("nvcc")
            .arg("--version")
            .output()
            .map_err(|e| ClientError::Hardware(format!("Failed to run nvcc: {}", e)))?;

        if !output.status.success() {
            return Err(ClientError::Hardware("nvcc failed".to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse "release X.Y" from nvcc output
        for line in stdout.lines() {
            if line.contains("release") {
                if let Some(version_part) = line.split("release").nth(1) {
                    if let Some(version) = version_part.split(',').next() {
                        return Ok(version.trim().to_string());
                    }
                }
            }
        }

        Err(ClientError::Hardware("Could not parse CUDA version".to_string()))
    }

    fn detect_cpu(&self) -> CpuInfo {
        let cpu_model = self.sys.global_cpu_info().brand().to_string();
        let cpu_cores = self.sys.cpus().len() as i32;
        let ram_bytes = self.sys.total_memory();
        let ram_gb = (ram_bytes / (1024 * 1024 * 1024)) as i32;

        CpuInfo {
            model: cpu_model,
            cores: cpu_cores,
            ram_gb,
        }
    }

    fn detect_os(&self) -> OsInfo {
        let name = format!(
            "{} {}",
            System::name().unwrap_or_else(|| "Unknown".to_string()),
            System::os_version().unwrap_or_else(|| "".to_string())
        );

        OsInfo { name }
    }

    fn estimate_bandwidth(&self) -> i32 {
        // This is a rough estimation - in production you'd want to run actual bandwidth tests
        // For now, return a conservative estimate
        1000 // 1 Gbps
    }
}

#[derive(Debug)]
struct GpuInfo {
    model: String,
    count: i32,
    vram_gb: i32,
    compute_capability: f32,
    driver_version: String,
    cuda_version: Option<String>,
}

#[derive(Debug)]
struct CpuInfo {
    model: String,
    cores: i32,
    ram_gb: i32,
}

#[derive(Debug)]
struct OsInfo {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detector_creation() {
        let detector = HardwareDetector::new();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_detect_cpu() {
        let detector = HardwareDetector::new().unwrap();
        let cpu_info = detector.detect_cpu();

        assert!(!cpu_info.model.is_empty());
        assert!(cpu_info.cores > 0);
        assert!(cpu_info.ram_gb > 0);
    }

    #[test]
    fn test_detect_os() {
        let detector = HardwareDetector::new().unwrap();
        let os_info = detector.detect_os();

        assert!(!os_info.name.is_empty());
    }
}
