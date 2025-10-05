# GLIN Provider

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

**Official reference implementation of the GLIN federated learning provider client.**

GLIN Provider is a CLI tool that allows GPU providers to participate in the GLIN federated learning marketplace. It handles provider registration, hardware detection, task polling, training execution, and gradient submission.

**[📚 Documentation →](https://docs.glin.ai/federated-learning/getting-started/train-model)** | **[🌐 Network Docs →](https://docs.glin.ai/network/intro)**

## 🚀 Quick Start

### Installation

```bash
# From source
git clone https://github.com/glin-ai/glin-provider
cd glin-provider
cargo install --path .
```

### Registration

```bash
# Register your GPU with the network
glin-provider register \
  --name "My GPU" \
  --wallet-address "0x..." \
  --backend-url "https://api.glin.ai"
```

### Start Providing Compute

```bash
# Start accepting federated learning tasks
glin-provider start
```

## 📋 Prerequisites

- **Hardware**: NVIDIA GPU with CUDA support (or AMD GPU with ROCm)
- **Software**:
  - Rust 1.70+
  - CUDA 11.0+ / ROCm 5.0+
  - Python 3.8+ with PyTorch or TensorFlow
  - 8GB+ available disk space

## ✨ Features

- 🚀 **Automatic Hardware Detection** - Detects GPU model, VRAM, CPU, and system specs
- 📊 **Performance Benchmarking** - Comprehensive benchmark suite for provider qualification
- 🔄 **Task Automation** - Automatic task polling and execution
- 💾 **IPFS Integration** - Decentralized model and gradient storage
- 🗜️ **Gradient Compression** - 4-10x compression with quantization/sparsification
- 📈 **Real-time Monitoring** - GPU usage, temperature, and progress tracking
- 🛡️ **Graceful Shutdown** - Safely completes active tasks before exit
- ⚡ **Concurrent Execution** - Run multiple training tasks in parallel

## 📖 Commands

### `register`
Register your provider with the GLIN network.

```bash
glin-provider register --name "My GPU" --wallet-address "0x..."
```

### `start`
Start the worker daemon to accept tasks.

```bash
glin-provider start
```

### `status`
Check your provider status and active tasks.

```bash
glin-provider status
```

### `benchmark`
Run GPU benchmark tests.

```bash
glin-provider benchmark
```

### `logs`
View worker logs.

```bash
glin-provider logs --tail 100
```

## 💡 Usage Examples

### Complete Setup Workflow

```bash
# 1. Install glin-provider
cargo install --path .

# 2. Run benchmark to check your hardware
glin-provider benchmark

# 3. Register with the network
glin-provider register \
  --name "My RTX 3090" \
  --wallet-address "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb" \
  --backend-url "https://api.glin.ai"

# 4. Start accepting tasks
glin-provider start

# 5. In another terminal, check status
glin-provider status
```

### Custom Configuration

```bash
# Register with custom availability and pricing
glin-provider register \
  --name "My GPU Farm" \
  --wallet-address "0x..." \
  --min-price-per-hour 5000 \
  --backend-url "https://api.glin.ai"

# Edit config manually
vim ~/.glin/config.toml

# Verify configuration
glin-provider status
```

### Running Benchmarks

```bash
# Quick benchmark (30 seconds)
glin-provider benchmark --quick

# Full benchmark suite (2-3 minutes)
glin-provider benchmark

# Benchmark output:
# Matrix Multiply Score: 85.2
# Gradient Compute Score: 82.7
# Memory Bandwidth Score: 78.9
# Overall Score: 83.1/100
```

### Monitoring Your Provider

```bash
# Check current status
glin-provider status

# View recent logs
glin-provider logs --tail 50

# Follow logs in real-time
glin-provider logs --follow

# Check configuration
cat ~/.glin/config.toml
```

### Development and Testing

```bash
# Use local backend for development
glin-provider register \
  --name "Dev GPU" \
  --wallet-address "0x123..." \
  --backend-url "http://localhost:3000"

# Start worker with debug logs
RUST_LOG=debug glin-provider start

# Run tests
cargo test

# Run integration tests only
cargo test --test integration_tests

# Run API tests only
cargo test --test api_tests
```

## ⚙️ Configuration

Configuration is stored in `~/.glin/config.toml`:

```toml
[provider]
name = "My GPU"
wallet_address = "0x..."
api_key = "..."
jwt_token = "..."

[backend]
url = "https://api.glin.ai"

[worker]
heartbeat_interval_secs = 30
task_poll_interval_secs = 10
max_concurrent_tasks = 1
```

## 🔒 Security

- API keys are stored securely in your local config
- All communication with the backend uses HTTPS
- JWT tokens are refreshed automatically
- Private keys never leave your machine

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Website**: https://glin.ai
- **Provider Docs**: https://docs.glin.ai/federated-learning/getting-started/train-model
- **Network Docs**: https://docs.glin.ai/network/intro
- **Full Documentation**: https://docs.glin.ai
- **Discord**: https://discord.gg/glin-ai
- **GitHub**: https://github.com/glin-ai
