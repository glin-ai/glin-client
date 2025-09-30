pub mod register;
pub mod start;
pub mod status;
pub mod benchmark;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "glin-client")]
#[command(about = "GLIN Federated Learning Provider CLI", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Register as a provider
    Register(register::RegisterArgs),

    /// Start the worker to accept tasks
    Start(start::StartArgs),

    /// Check provider status
    Status(status::StatusArgs),

    /// Run performance benchmark
    Benchmark(benchmark::BenchmarkArgs),
}
