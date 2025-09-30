use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterProviderRequest {
    pub name: String,
    pub wallet_address: String,
    pub hardware_info: HardwareInfo,
    pub availability_hours: Vec<AvailabilityWindow>,
    pub min_price_per_hour: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub gpu_model: String,
    pub gpu_count: i32,
    pub vram_gb: i32,
    pub compute_capability: f32,
    pub cpu_model: String,
    pub cpu_cores: i32,
    pub ram_gb: i32,
    pub bandwidth_mbps: i32,
    pub os: String,
    pub driver_version: String,
    pub cuda_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityWindow {
    pub day_of_week: u8,
    pub start_hour: u8,
    pub end_hour: u8,
    pub timezone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHeartbeat {
    pub provider_id: Uuid,
    pub current_tasks: Vec<Uuid>,
    pub cpu_usage: f32,
    pub gpu_usage: f32,
    pub memory_usage: f32,
    pub temperature: f32,
    pub available_vram_gb: f32,
}

// Response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterProviderResponse {
    pub provider: Provider,
    pub api_key: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub id: Uuid,
    pub name: String,
    pub wallet_address: String,
    pub reputation_score: f64,
    pub total_tasks_completed: i32,
    pub total_gradients_computed: i64,
    pub total_tokens_earned: i64,
    pub status: ProviderStatus,
    pub last_heartbeat: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderStatus {
    Active,
    Idle,
    Busy,
    Offline,
    Suspended,
    Banned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderTaskInfo {
    pub id: Uuid,
    pub name: String,
    pub task_status: String,
    pub batch_start: i32,
    pub batch_end: i32,
    pub assignment_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitGradientRequest {
    pub task_id: Uuid,
    pub provider_id: Uuid,
    pub gradient_cid: String,
    pub metrics: GradientMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub training_duration_secs: u64,
    pub compression_method: String,
}
