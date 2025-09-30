use clap::Args;

use crate::{
    config::Config,
    error::Result,
    worker::Worker,
};

#[derive(Args)]
pub struct StartArgs {
    /// Run as daemon (background process)
    #[arg(short, long)]
    pub daemon: bool,
}

pub async fn execute(args: StartArgs) -> Result<()> {
    // Load configuration
    let config = Config::load()?;

    if !config.is_registered() {
        return Err(crate::error::ClientError::NotRegistered);
    }

    println!("🚀 Starting GLIN worker...");
    println!("Provider ID: {}", config.provider.id.unwrap());
    println!("Backend: {}", config.backend.url);
    println!("\nPress Ctrl+C to stop\n");

    // Create and start worker
    let worker = Worker::new(config).await?;
    worker.run().await?;

    Ok(())
}
