use glin_client::{
    api::{BackendClient, RegisterProviderRequest, HardwareInfo, AvailabilityWindow},
    config::Config,
    gpu::{HardwareDetector, Benchmarker},
};
use tempfile::TempDir;

#[tokio::test]
async fn test_config_save_and_load() {
    let temp_dir = TempDir::new().unwrap();
    std::env::set_var("HOME", temp_dir.path());

    let mut config = Config::default();
    config.provider.name = "Test Provider".to_string();
    config.provider.wallet_address = "0x123...".to_string();
    config.backend.url = "http://localhost:3000".to_string();

    // Save config
    config.save().unwrap();

    // Load config
    let loaded_config = Config::load().unwrap();
    assert_eq!(loaded_config.provider.name, "Test Provider");
    assert_eq!(loaded_config.backend.url, "http://localhost:3000");
}

#[test]
fn test_hardware_detector() {
    let detector = HardwareDetector::new().unwrap();
    let hardware_info = detector.detect().unwrap();

    // CPU model might be empty on some systems, so just check cores and RAM
    assert!(hardware_info.cpu_cores > 0);
    assert!(hardware_info.ram_gb > 0);
    assert!(!hardware_info.os.is_empty());
}

#[test]
fn test_benchmarker() {
    let benchmarker = Benchmarker::new().unwrap();
    let results = benchmarker.run_full_benchmark().unwrap();

    assert!(results.matrix_multiply_score >= 0.0);
    assert!(results.gradient_compute_score >= 0.0);
    assert!(results.memory_bandwidth_score >= 0.0);
    assert!(results.overall_score >= 0.0);
    assert!(results.overall_score <= 100.0);
}

#[test]
fn test_quick_benchmark() {
    let benchmarker = Benchmarker::new().unwrap();
    let results = benchmarker.run_quick_benchmark().unwrap();

    assert!(results.overall_score >= 0.0);
    assert!(results.overall_score <= 100.0);
    assert!(results.execution_time_ms < 1000); // Quick should be fast
}

#[tokio::test]
async fn test_backend_client_creation() {
    let mut client = BackendClient::new("http://localhost:3000");
    client.set_token("test_token");

    // Client should be created successfully
    // Actual API calls tested in api_tests.rs
}

#[test]
fn test_register_provider_request_serialization() {
    let request = RegisterProviderRequest {
        name: "Test".to_string(),
        wallet_address: "0x123".to_string(),
        hardware_info: HardwareInfo {
            gpu_model: "NVIDIA RTX 3090".to_string(),
            gpu_count: 1,
            vram_gb: 24,
            compute_capability: 8.6,
            cpu_model: "Intel i9".to_string(),
            cpu_cores: 16,
            ram_gb: 64,
            bandwidth_mbps: 1000,
            os: "Linux".to_string(),
            driver_version: "535.54.03".to_string(),
            cuda_version: Some("12.1".to_string()),
        },
        availability_hours: vec![
            AvailabilityWindow {
                day_of_week: 1,
                start_hour: 9,
                end_hour: 17,
                timezone: "UTC".to_string(),
            },
        ],
        min_price_per_hour: 1000,
    };

    // Test serialization
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("Test"));
    assert!(json.contains("RTX 3090"));

    // Test deserialization
    let deserialized: RegisterProviderRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "Test");
    assert_eq!(deserialized.hardware_info.gpu_count, 1);
}

#[tokio::test]
async fn test_config_validation() {
    let config = Config::default();

    assert!(!config.is_registered());
    assert!(config.provider.id.is_none());
}

#[test]
fn test_error_types() {
    use glin_client::error::ClientError;

    let config_error = ClientError::Config("test error".to_string());
    assert_eq!(config_error.to_string(), "Configuration error: test error");

    let api_error = ClientError::Api("API failed".to_string());
    assert_eq!(api_error.to_string(), "API error: API failed");

    let not_registered = ClientError::NotRegistered;
    assert!(not_registered.to_string().contains("register"));
}

#[test]
fn test_availability_window() {
    let window = AvailabilityWindow {
        day_of_week: 1, // Monday
        start_hour: 9,
        end_hour: 17,
        timezone: "UTC".to_string(),
    };

    assert_eq!(window.day_of_week, 1);
    assert!(window.start_hour < window.end_hour);
}

#[tokio::test]
async fn test_end_to_end_workflow_simulation() {
    // Simulate the workflow without actual backend

    // 1. Hardware detection
    let detector = HardwareDetector::new().unwrap();
    let hw_info = detector.detect().unwrap();
    assert!(hw_info.cpu_cores > 0);

    // 2. Benchmark
    let benchmarker = Benchmarker::new().unwrap();
    let bench_results = benchmarker.run_quick_benchmark().unwrap();
    assert!(bench_results.overall_score > 0.0);

    // 3. Config management
    let temp_dir = TempDir::new().unwrap();
    std::env::set_var("HOME", temp_dir.path());

    let mut config = Config::default();
    config.provider.name = "Test Provider".to_string();
    config.provider.wallet_address = "0x123".to_string();
    config.save().unwrap();

    let loaded = Config::load().unwrap();
    assert_eq!(loaded.provider.name, "Test Provider");
}
