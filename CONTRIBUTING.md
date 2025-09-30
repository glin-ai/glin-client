# Contributing to GLIN Client

Thank you for your interest in contributing to GLIN Client! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. By participating in this project, you agree to:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- Rust 1.70 or later installed
- Git for version control
- A GitHub account
- Familiarity with Rust and command-line tools

### Finding Issues to Work On

- Check the [issue tracker](https://github.com/glin-ai/glin-client/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Comment on an issue to let others know you're working on it
- Ask questions if anything is unclear

## 💻 Development Setup

1. **Fork the repository:**

Visit https://github.com/glin-ai/glin-client and click "Fork"

2. **Clone your fork:**

```bash
git clone https://github.com/YOUR_USERNAME/glin-client.git
cd glin-client
```

3. **Add the upstream remote:**

```bash
git remote add upstream https://github.com/glin-ai/glin-client.git
```

4. **Install dependencies:**

```bash
# Rust dependencies (automatically handled by Cargo)
cargo build

# Python dependencies
pip install -r python/requirements.txt
```

5. **Run tests to verify setup:**

```bash
cargo test
```

## 🔨 Making Changes

### Creating a Branch

Always create a new branch for your changes:

```bash
git checkout -b feature/my-new-feature
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or changes

### Making Commits

Write clear, concise commit messages:

```bash
git commit -m "feat: add gradient compression support

- Implement quantization algorithm
- Add sparsification method
- Update documentation"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation change
- `test:` - Test addition/modification
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### Keeping Your Fork Updated

Regularly sync with the upstream repository:

```bash
git fetch upstream
git rebase upstream/main
```

## 🧪 Testing

### Running Tests

```bash
# All tests
cargo test

# Specific test file
cargo test --test integration_tests

# With logging
RUST_LOG=debug cargo test

# Run benchmarks
cargo bench
```

### Writing Tests

- Add unit tests in the same file as the code
- Add integration tests in `tests/` directory
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

Example:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        let result = my_function();
        assert_eq!(result, expected_value);
    }
}
```

### Python Tests

```bash
# Run Python tests
python -m pytest python/tests/

# With coverage
python -m pytest --cov=python python/tests/
```

## 📤 Submitting Changes

### Before Submitting

1. **Run all tests:**
```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

2. **Update documentation:**
- Update README.md if adding features
- Add inline documentation for new functions
- Update CHANGELOG.md

3. **Check code quality:**
```bash
cargo clippy -- -D warnings
```

### Creating a Pull Request

1. **Push your changes:**
```bash
git push origin feature/my-new-feature
```

2. **Open a Pull Request on GitHub:**
- Provide a clear title and description
- Reference related issues (e.g., "Fixes #123")
- Describe what changed and why
- Include screenshots for UI changes
- List any breaking changes

3. **PR Template:**

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

- Maintainers will review your PR
- Address feedback and comments
- Make requested changes
- Once approved, your PR will be merged

## 📝 Style Guidelines

### Rust Code Style

Follow the official [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/):

```rust
// Use snake_case for functions and variables
fn calculate_gradient() -> f64 {
    let learning_rate = 0.001;
    // ...
}

// Use CamelCase for types
struct TrainingConfig {
    epochs: u32,
    batch_size: u32,
}

// Document public APIs
/// Computes the gradient for the given model.
///
/// # Arguments
/// * `model` - The neural network model
/// * `data` - Training data
///
/// # Returns
/// Computed gradients as a dictionary
pub fn compute_gradient(model: &Model, data: &Data) -> Gradients {
    // ...
}
```

### Python Code Style

Follow [PEP 8](https://peps.python.org/pep-0008/):

```python
# Use snake_case for functions and variables
def calculate_gradient(model, data):
    learning_rate = 0.001
    # ...

# Use docstrings
def train_model(model, dataset, epochs=10):
    """
    Train a PyTorch model on the given dataset.

    Args:
        model: PyTorch model to train
        dataset: Training dataset
        epochs: Number of training epochs (default: 10)

    Returns:
        Training metrics including loss and accuracy
    """
    # ...
```

### Documentation

- Use clear, concise language
- Provide code examples
- Explain the "why," not just the "what"
- Keep documentation up to date with code changes

## 🏗️ Project Structure

```
glin-client/
├── src/
│   ├── api/          # Backend API client
│   ├── cli/          # CLI commands
│   ├── config/       # Configuration management
│   ├── gpu/          # GPU detection and benchmarking
│   ├── storage/      # IPFS and caching
│   ├── worker/       # Task execution
│   └── main.rs       # Entry point
├── python/
│   ├── train.py      # Training script
│   └── utils/        # Gradient and compression utilities
├── tests/
│   ├── integration_tests.rs
│   └── api_tests.rs
└── docs/             # Additional documentation
```

## 🐛 Reporting Bugs

When reporting bugs, include:

- **Environment details**: OS, Rust version, GPU model
- **Steps to reproduce**: Clear, minimal reproduction steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages and logs
- **Screenshots**: If applicable

Use the bug report template in GitHub Issues.

## 💡 Suggesting Enhancements

When suggesting features:

- Check if it's already been suggested
- Clearly describe the feature and use case
- Explain why it would be useful
- Provide examples if possible

Use the feature request template in GitHub Issues.

## 🤝 Community

### Getting Help

- **Discord**: https://discord.gg/glin
- **GitHub Discussions**: https://github.com/glin-ai/glin-client/discussions
- **Email**: dev@glin.ai

### Stay Updated

- Watch the repository for updates
- Join our Discord for discussions
- Follow our blog for announcements

## 📄 License

By contributing to GLIN Client, you agree that your contributions will be licensed under the Apache 2.0 License.

## 🙏 Thank You!

Your contributions make GLIN Client better for everyone. We appreciate your time and effort!

---

**Questions?** Open an issue or ask in our Discord channel.
