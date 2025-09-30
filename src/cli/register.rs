use clap::Args;

use crate::{
    api::BackendClient,
    config::Config,
    error::Result,
    gpu::HardwareDetector,
};

#[derive(Args)]
pub struct RegisterArgs {
    /// Provider name
    #[arg(short, long)]
    pub name: String,

    /// Wallet address for rewards
    #[arg(short, long)]
    pub wallet_address: String,

    /// Backend API URL
    #[arg(short, long, default_value = "http://localhost:3000")]
    pub backend_url: String,
}

pub async fn execute(args: RegisterArgs) -> Result<()> {
    tracing::info!("Starting provider registration...");

    // Check if already registered
    if let Ok(config) = Config::load() {
        if config.is_registered() {
            return Err(crate::error::ClientError::AlreadyRegistered(
                config.provider.id.unwrap().to_string(),
            ));
        }
    }

    // Detect hardware
    tracing::info!("Detecting hardware...");
    let hardware_detector = HardwareDetector::new()?;
    let hardware_info = hardware_detector.detect()?;

    tracing::info!("GPU: {} with {}GB VRAM", hardware_info.gpu_model, hardware_info.vram_gb);
    tracing::info!("CPU: {} ({} cores)", hardware_info.cpu_model, hardware_info.cpu_cores);

    // Register with backend
    tracing::info!("Registering with backend at {}...", args.backend_url);
    let client = BackendClient::new(&args.backend_url);

    let registration = crate::api::types::RegisterProviderRequest {
        name: args.name.clone(),
        wallet_address: args.wallet_address.clone(),
        hardware_info,
        availability_hours: vec![], // TODO: Let user configure
        min_price_per_hour: 1000,  // TODO: Let user configure
    };

    let response = client.register(registration).await?;

    // Save configuration
    let mut config = Config::default();
    config.provider.id = Some(response.provider.id);
    config.provider.name = args.name;
    config.provider.wallet_address = args.wallet_address;
    config.provider.api_key = Some(response.api_key);
    config.provider.jwt_token = Some(response.token);
    config.backend.url = args.backend_url;

    config.save()?;

    println!("✅ Registration successful!");
    println!("Provider ID: {}", response.provider.id);
    println!("API Key saved to config");
    println!("\nYou can now start accepting tasks with:");
    println!("  glin-client start");

    Ok(())
}
