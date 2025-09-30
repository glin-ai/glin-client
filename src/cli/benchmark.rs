use clap::Args;

use crate::{
    error::Result,
    gpu::Benchmarker,
};

#[derive(Args)]
pub struct BenchmarkArgs {
    /// Run quick benchmark (faster but less accurate)
    #[arg(short, long)]
    pub quick: bool,
}

pub async fn execute(args: BenchmarkArgs) -> Result<()> {
    println!("🚀 Starting GPU benchmark...\n");

    let benchmarker = Benchmarker::new()?;

    let results = if args.quick {
        benchmarker.run_quick_benchmark()?
    } else {
        benchmarker.run_full_benchmark()?
    };

    results.print_summary();

    println!("\n💡 Tip: Use --quick flag for faster benchmarks during testing");

    Ok(())
}
