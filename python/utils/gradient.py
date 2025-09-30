"""
Gradient computation and extraction utilities for federated learning.
"""

import torch
import torch.nn as nn


def extract_gradients(model: nn.Module) -> dict:
    """
    Extract gradients from a PyTorch model.

    Args:
        model: Trained PyTorch model with computed gradients

    Returns:
        Dictionary mapping parameter names to gradient tensors
    """
    gradients = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Clone to detach from computational graph
            gradients[name] = param.grad.cpu().clone()
        else:
            # If no gradient computed, store zero tensor
            gradients[name] = torch.zeros_like(param.cpu())

    return gradients


def compute_gradient_norm(gradients: dict) -> float:
    """
    Compute the L2 norm of all gradients.

    Args:
        gradients: Dictionary of gradient tensors

    Returns:
        L2 norm of all gradients combined
    """
    total_norm = 0.0

    for grad in gradients.values():
        total_norm += grad.norm(2).item() ** 2

    return total_norm ** 0.5


def clip_gradients(gradients: dict, max_norm: float = 1.0) -> dict:
    """
    Clip gradients to a maximum norm (for stability).

    Args:
        gradients: Dictionary of gradient tensors
        max_norm: Maximum allowed gradient norm

    Returns:
        Dictionary of clipped gradients
    """
    current_norm = compute_gradient_norm(gradients)

    if current_norm > max_norm:
        clip_coef = max_norm / (current_norm + 1e-6)
        clipped_gradients = {}

        for name, grad in gradients.items():
            clipped_gradients[name] = grad * clip_coef

        return clipped_gradients

    return gradients


def add_noise_to_gradients(gradients: dict, noise_scale: float = 0.01) -> dict:
    """
    Add Gaussian noise to gradients for differential privacy.

    Args:
        gradients: Dictionary of gradient tensors
        noise_scale: Standard deviation of noise to add

    Returns:
        Dictionary of noisy gradients
    """
    noisy_gradients = {}

    for name, grad in gradients.items():
        noise = torch.randn_like(grad) * noise_scale
        noisy_gradients[name] = grad + noise

    return noisy_gradients


def aggregate_gradients(gradient_list: list) -> dict:
    """
    Aggregate multiple gradient dictionaries (e.g., from multiple batches).

    Args:
        gradient_list: List of gradient dictionaries

    Returns:
        Averaged gradient dictionary
    """
    if not gradient_list:
        return {}

    # Initialize with zeros
    aggregated = {name: torch.zeros_like(grad)
                  for name, grad in gradient_list[0].items()}

    # Sum all gradients
    for gradients in gradient_list:
        for name, grad in gradients.items():
            aggregated[name] += grad

    # Average
    num_gradients = len(gradient_list)
    for name in aggregated:
        aggregated[name] /= num_gradients

    return aggregated


def save_gradients(gradients: dict, output_path: str, compress: bool = False):
    """
    Save gradients to a file.

    Args:
        gradients: Dictionary of gradient tensors
        output_path: Path to save the gradients
        compress: Whether to compress the gradients
    """
    if compress:
        # Save with compression
        torch.save(gradients, output_path, _use_new_zipfile_serialization=True)
    else:
        torch.save(gradients, output_path)


def load_gradients(gradient_path: str) -> dict:
    """
    Load gradients from a file.

    Args:
        gradient_path: Path to the gradient file

    Returns:
        Dictionary of gradient tensors
    """
    return torch.load(gradient_path, map_location='cpu')


def gradient_statistics(gradients: dict) -> dict:
    """
    Compute statistics about gradients for logging/debugging.

    Args:
        gradients: Dictionary of gradient tensors

    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_parameters': len(gradients),
        'total_elements': sum(grad.numel() for grad in gradients.values()),
        'l2_norm': compute_gradient_norm(gradients),
        'max_value': max(grad.abs().max().item() for grad in gradients.values()),
        'min_value': min(grad.abs().min().item() for grad in gradients.values()),
        'mean_value': sum(grad.mean().item() for grad in gradients.values()) / len(gradients),
    }

    return stats
