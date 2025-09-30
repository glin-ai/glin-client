use clap::Args;

use crate::{
    api::BackendClient,
    config::Config,
    error::Result,
    worker::GpuMonitor,
};

#[derive(Args)]
pub struct StatusArgs {}

pub async fn execute(_args: StatusArgs) -> Result<()> {
    // Load configuration
    let config = Config::load()?;

    if !config.is_registered() {
        return Err(crate::error::ClientError::NotRegistered);
    }

    println!("📊 Provider Status\n");
    println!("Provider ID: {}", config.provider.id.unwrap());
    println!("Name: {}", config.provider.name);
    println!("Wallet: {}", config.provider.wallet_address);
    println!("Backend: {}", config.backend.url);

    // Get provider details from backend
    let mut client = BackendClient::new(&config.backend.url);
    client.set_token(config.provider.jwt_token.as_ref().unwrap());

    match client.get_provider(config.provider.id.unwrap()).await {
        Ok(provider) => {
            println!("\n📈 Statistics");
            println!("Reputation: {:.2}", provider.reputation_score);
            println!("Tasks Completed: {}", provider.total_tasks_completed);
            println!("Tokens Earned: {}", provider.total_tokens_earned);
            println!("Status: {:?}", provider.status);
        }
        Err(e) => {
            tracing::warn!("Could not fetch provider details: {}", e);
        }
    }

    // Get active tasks
    match client.get_provider_tasks().await {
        Ok(tasks) => {
            if tasks.is_empty() {
                println!("\n📋 No active tasks");
            } else {
                println!("\n📋 Active Tasks ({}):", tasks.len());
                for task in tasks {
                    println!("  - Task: {}", task.name);
                    println!("    Status: {:?}", task.task_status);
                }
            }
        }
        Err(e) => {
            tracing::warn!("Could not fetch tasks: {}", e);
        }
    }

    // Show GPU stats
    let gpu_monitor = GpuMonitor::new();
    println!("\n🖥️  GPU Status");
    println!("GPU Usage: {:.1}%", gpu_monitor.get_usage());
    println!("Memory Usage: {:.1}%", gpu_monitor.get_memory_usage());
    println!("Temperature: {:.1}°C", gpu_monitor.get_temperature());
    println!("Available VRAM: {:.1} GB", gpu_monitor.get_available_vram());

    Ok(())
}
