use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod cli;
mod config;
mod error;
mod gpu;
mod storage;
mod worker;

use cli::{Cli, Commands};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "glin_provider=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Register(args) => cli::register::execute(args).await,
        Commands::Start(args) => cli::start::execute(args).await,
        Commands::Status(args) => cli::status::execute(args).await,
        Commands::Benchmark(args) => cli::benchmark::execute(args).await,
    };

    if let Err(e) = result {
        eprintln!("❌ Error: {}", e);
        std::process::exit(1);
    }
}
