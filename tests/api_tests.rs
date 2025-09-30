use glin_client::api::{
    BackendClient, RegisterProviderRequest, HardwareInfo, ProviderHeartbeat,
    AvailabilityWindow, SubmitGradientRequest, GradientMetrics,
};
use mockito::Server;
use uuid::Uuid;

#[tokio::test]
async fn test_register_provider_success() {
    let mut server = Server::new_async().await;

    let mock = server
        .mock("POST", "/api/v1/providers/register")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{
            "provider": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Test Provider",
                "wallet_address": "0x123",
                "reputation_score": 0.0,
                "total_tasks_completed": 0,
                "total_gradients_computed": 0,
                "total_tokens_earned": 0,
                "status": "active",
                "last_heartbeat": "2024-01-01T00:00:00Z",
                "created_at": "2024-01-01T00:00:00Z"
            },
            "api_key": "test_api_key",
            "token": "test_jwt_token"
        }"#)
        .create_async()
        .await;

    let client = BackendClient::new(&server.url());

    let request = RegisterProviderRequest {
        name: "Test Provider".to_string(),
        wallet_address: "0x123".to_string(),
        hardware_info: HardwareInfo {
            gpu_model: "Test GPU".to_string(),
            gpu_count: 1,
            vram_gb: 8,
            compute_capability: 7.5,
            cpu_model: "Test CPU".to_string(),
            cpu_cores: 8,
            ram_gb: 16,
            bandwidth_mbps: 1000,
            os: "Linux".to_string(),
            driver_version: "1.0".to_string(),
            cuda_version: Some("11.0".to_string()),
        },
        availability_hours: vec![],
        min_price_per_hour: 1000,
    };

    let response = client.register(request).await;
    assert!(response.is_ok());

    let response = response.unwrap();
    assert_eq!(response.provider.name, "Test Provider");
    assert_eq!(response.api_key, "test_api_key");
    assert_eq!(response.token, "test_jwt_token");

    mock.assert_async().await;
}

#[tokio::test]
async fn test_register_provider_failure() {
    let mut server = Server::new_async().await;

    let _mock = server
        .mock("POST", "/api/v1/providers/register")
        .with_status(400)
        .with_body("Invalid request")
        .create_async()
        .await;

    let client = BackendClient::new(&server.url());

    let request = RegisterProviderRequest {
        name: "Test".to_string(),
        wallet_address: "0x123".to_string(),
        hardware_info: HardwareInfo {
            gpu_model: "Test".to_string(),
            gpu_count: 1,
            vram_gb: 8,
            compute_capability: 7.5,
            cpu_model: "Test".to_string(),
            cpu_cores: 8,
            ram_gb: 16,
            bandwidth_mbps: 1000,
            os: "Linux".to_string(),
            driver_version: "1.0".to_string(),
            cuda_version: None,
        },
        availability_hours: vec![],
        min_price_per_hour: 1000,
    };

    let response = client.register(request).await;
    assert!(response.is_err());
}

#[tokio::test]
async fn test_heartbeat_success() {
    let mut server = Server::new_async().await;

    let mock = server
        .mock("POST", "/api/v1/providers/heartbeat")
        .match_header("authorization", "Bearer test_token")
        .with_status(200)
        .create_async()
        .await;

    let mut client = BackendClient::new(&server.url());
    client.set_token("test_token");

    let heartbeat = ProviderHeartbeat {
        provider_id: Uuid::new_v4(),
        current_tasks: vec![],
        cpu_usage: 50.0,
        gpu_usage: 60.0,
        memory_usage: 70.0,
        temperature: 65.0,
        available_vram_gb: 8.0,
    };

    let response = client.heartbeat(heartbeat).await;
    assert!(response.is_ok());

    mock.assert_async().await;
}

#[tokio::test]
async fn test_get_provider_tasks_success() {
    let mut server = Server::new_async().await;

    let mock = server
        .mock("GET", "/api/v1/providers/tasks")
        .match_header("authorization", "Bearer test_token")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"[
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Test Task",
                "task_status": "assigned",
                "batch_start": 0,
                "batch_end": 100,
                "assignment_status": "active"
            }
        ]"#)
        .create_async()
        .await;

    let mut client = BackendClient::new(&server.url());
    client.set_token("test_token");

    let tasks = client.get_provider_tasks().await;
    assert!(tasks.is_ok());

    let tasks = tasks.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].name, "Test Task");

    mock.assert_async().await;
}

#[tokio::test]
async fn test_get_provider_tasks_empty() {
    let mut server = Server::new_async().await;

    let mock = server
        .mock("GET", "/api/v1/providers/tasks")
        .match_header("authorization", "Bearer test_token")
        .with_status(404)
        .create_async()
        .await;

    let mut client = BackendClient::new(&server.url());
    client.set_token("test_token");

    let tasks = client.get_provider_tasks().await;
    assert!(tasks.is_ok());
    assert_eq!(tasks.unwrap().len(), 0);

    mock.assert_async().await;
}

#[tokio::test]
async fn test_submit_gradient_success() {
    let mut server = Server::new_async().await;

    let mock = server
        .mock("POST", "/api/v1/gradients/submit")
        .match_header("authorization", "Bearer test_token")
        .with_status(200)
        .create_async()
        .await;

    let mut client = BackendClient::new(&server.url());
    client.set_token("test_token");

    let request = SubmitGradientRequest {
        task_id: Uuid::new_v4(),
        provider_id: Uuid::new_v4(),
        gradient_cid: "QmTest123".to_string(),
        metrics: GradientMetrics {
            loss: 0.5,
            accuracy: 0.95,
            training_duration_secs: 60,
            compression_method: "quantize".to_string(),
        },
    };

    let response = client.submit_gradient(request).await;
    assert!(response.is_ok());

    mock.assert_async().await;
}

#[test]
fn test_heartbeat_serialization() {
    let heartbeat = ProviderHeartbeat {
        provider_id: Uuid::new_v4(),
        current_tasks: vec![Uuid::new_v4()],
        cpu_usage: 50.0,
        gpu_usage: 75.0,
        memory_usage: 60.0,
        temperature: 70.0,
        available_vram_gb: 10.0,
    };

    let json = serde_json::to_string(&heartbeat).unwrap();
    assert!(json.contains("cpu_usage"));
    assert!(json.contains("50"));

    let deserialized: ProviderHeartbeat = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.cpu_usage, 50.0);
}

#[test]
fn test_gradient_metrics_serialization() {
    let metrics = GradientMetrics {
        loss: 0.123,
        accuracy: 0.987,
        training_duration_secs: 120,
        compression_method: "quantize".to_string(),
    };

    let json = serde_json::to_string(&metrics).unwrap();
    assert!(json.contains("loss"));
    assert!(json.contains("0.123"));
    assert!(json.contains("quantize"));

    let deserialized: GradientMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.loss, 0.123);
    assert_eq!(deserialized.compression_method, "quantize");
}
