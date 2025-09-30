#!/usr/bin/env python3
"""
GLIN Federated Learning Training Script

This script performs training on a model and dataset, then outputs gradients.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch not installed. Please run: pip install torch torchvision")
    sys.exit(1)

# Import gradient and compression utilities
try:
    from utils.gradient import extract_gradients, save_gradients, gradient_statistics, clip_gradients
    from utils.compression import compress_gradients, estimate_compression_ratio
except ImportError:
    print("WARNING: Could not import gradient utilities. Using basic implementation.")
    extract_gradients = None


def parse_args():
    parser = argparse.ArgumentParser(description='GLIN Training Script')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--compress', type=str, default='quantize',
                       choices=['none', 'quantize', 'sparsify'],
                       help='Gradient compression method')
    parser.add_argument('--clip-norm', type=float, default=None,
                       help='Gradient clipping norm (optional)')
    return parser.parse_args()


def load_model(model_path):
    """Load model from file or create a simple model"""
    print(f"Loading model from {model_path}")

    try:
        model = torch.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Creating simple dummy model for demonstration")

        # Simple neural network for demonstration
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 128)
                self.fc2 = nn.Linear(128, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = x.view(-1, 784)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return SimpleModel()


def load_dataset(dataset_path):
    """Load dataset or create dummy data"""
    print(f"Loading dataset from {dataset_path}")

    # For demonstration, create dummy data
    # In production, this would load real data
    print("Creating dummy dataset for demonstration")

    # Dummy MNIST-like data
    X = torch.randn(100, 1, 28, 28)  # 100 samples, 28x28 images
    y = torch.randint(0, 10, (100,))  # 100 labels (0-9)

    return X, y


def train_model(model, data, labels, epochs, batch_size, learning_rate):
    """Train the model and return gradients"""
    print(f"\nStarting training:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Mini-batch training
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_data = data[start_idx:end_idx].to(device)
            batch_labels = labels[start_idx:end_idx].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += batch_labels.size(0)
            epoch_correct += (predicted == batch_labels).sum().item()

        avg_loss = epoch_loss / num_batches
        accuracy = 100 * epoch_correct / epoch_total

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        total_loss += avg_loss
        correct += epoch_correct
        total += epoch_total

    final_loss = total_loss / epochs
    final_accuracy = 100 * correct / total

    print(f"\nTraining complete!")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")

    # Extract gradients using utility function if available
    if extract_gradients is not None:
        gradients = extract_gradients(model)

        # Print gradient statistics
        stats = gradient_statistics(gradients)
        print(f"\nGradient Statistics:")
        print(f"  Number of parameters: {stats['num_parameters']}")
        print(f"  Total elements: {stats['total_elements']}")
        print(f"  L2 norm: {stats['l2_norm']:.6f}")
        print(f"  Max absolute value: {stats['max_value']:.6f}")
        print(f"  Mean value: {stats['mean_value']:.6f}")
    else:
        # Fallback: basic gradient extraction
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()

    return {
        'loss': final_loss,
        'accuracy': final_accuracy,
        'gradients': gradients
    }


def save_results(output_dir, results, compression_method='quantize', clip_norm=None):
    """Save training results with optional compression"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gradients = results['gradients']

    # Apply gradient clipping if requested
    if clip_norm is not None and extract_gradients is not None:
        print(f"\nApplying gradient clipping (max norm: {clip_norm})")
        gradients = clip_gradients(gradients, max_norm=clip_norm)

    # Compress gradients if compression utilities available
    if compression_method != 'none' and extract_gradients is not None:
        print(f"\nCompressing gradients using method: {compression_method}")

        if compression_method == 'quantize':
            compressed = compress_gradients(gradients, method='quantize', num_bits=8)
        elif compression_method == 'sparsify':
            compressed = compress_gradients(gradients, method='sparsify', sparsity=0.9)
        else:
            compressed = {'gradients': gradients, 'method': 'none'}

        # Calculate compression ratio
        ratio = estimate_compression_ratio(gradients, compressed)
        print(f"Compression ratio: {ratio:.2f}x")

        # Save compressed gradients
        gradients_path = output_path / "gradients.pt"
        torch.save(compressed, gradients_path)
        print(f"Compressed gradients saved to {gradients_path}")
    else:
        # Save uncompressed gradients
        gradients_path = output_path / "gradients.pt"
        torch.save(gradients, gradients_path)
        print(f"\nGradients saved to {gradients_path}")

    # Save metrics
    metrics = {
        'loss': results['loss'],
        'accuracy': results['accuracy'],
        'compression_method': compression_method,
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("GLIN Federated Learning Training")
    print("=" * 60)

    # Load model
    model = load_model(args.model)

    # Load dataset
    data, labels = load_dataset(args.dataset)

    # Train model
    results = train_model(
        model, data, labels,
        args.epochs, args.batch_size, args.learning_rate
    )

    # Save results with compression
    save_results(args.output, results, args.compress, args.clip_norm)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
