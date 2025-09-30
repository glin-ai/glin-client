"""Utility modules for gradient computation and compression."""

from .gradient import (
    extract_gradients,
    compute_gradient_norm,
    clip_gradients,
    add_noise_to_gradients,
    aggregate_gradients,
    save_gradients,
    load_gradients,
    gradient_statistics,
)

from .compression import (
    quantize_gradients,
    dequantize_gradients,
    sparsify_gradients,
    densify_gradients,
    compress_gradients,
    decompress_gradients,
    estimate_compression_ratio,
)

__all__ = [
    # Gradient utilities
    'extract_gradients',
    'compute_gradient_norm',
    'clip_gradients',
    'add_noise_to_gradients',
    'aggregate_gradients',
    'save_gradients',
    'load_gradients',
    'gradient_statistics',
    # Compression utilities
    'quantize_gradients',
    'dequantize_gradients',
    'sparsify_gradients',
    'densify_gradients',
    'compress_gradients',
    'decompress_gradients',
    'estimate_compression_ratio',
]
