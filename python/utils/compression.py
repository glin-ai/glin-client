"""
Gradient compression utilities for efficient transmission.
"""

import torch


def quantize_gradients(gradients: dict, num_bits: int = 8) -> dict:
    """
    Quantize gradients to reduce size.

    Args:
        gradients: Dictionary of gradient tensors
        num_bits: Number of bits for quantization (e.g., 8 for int8)

    Returns:
        Dictionary of quantized gradients
    """
    quantized = {}
    quantization_params = {}

    for name, grad in gradients.items():
        # Find min and max values
        min_val = grad.min().item()
        max_val = grad.max().item()

        # Calculate scale and zero point
        scale = (max_val - min_val) / (2 ** num_bits - 1)
        zero_point = min_val

        # Quantize
        if scale > 0:
            quantized_grad = ((grad - zero_point) / scale).round().clamp(0, 2 ** num_bits - 1)
            quantized[name] = quantized_grad.to(torch.uint8)
            quantization_params[name] = {'scale': scale, 'zero_point': zero_point}
        else:
            # If all values are the same, just store zeros
            quantized[name] = torch.zeros_like(grad, dtype=torch.uint8)
            quantization_params[name] = {'scale': 0.0, 'zero_point': min_val}

    return {'gradients': quantized, 'params': quantization_params}


def dequantize_gradients(quantized_data: dict) -> dict:
    """
    Dequantize gradients back to float32.

    Args:
        quantized_data: Dictionary with 'gradients' and 'params' keys

    Returns:
        Dictionary of dequantized gradients
    """
    quantized = quantized_data['gradients']
    params = quantized_data['params']
    dequantized = {}

    for name, quantized_grad in quantized.items():
        scale = params[name]['scale']
        zero_point = params[name]['zero_point']

        # Dequantize
        dequantized_grad = quantized_grad.float() * scale + zero_point
        dequantized[name] = dequantized_grad

    return dequantized


def sparsify_gradients(gradients: dict, sparsity: float = 0.9) -> dict:
    """
    Sparsify gradients by keeping only top-k values.

    Args:
        gradients: Dictionary of gradient tensors
        sparsity: Fraction of values to zero out (0.9 = keep top 10%)

    Returns:
        Dictionary of sparse gradients with indices
    """
    sparse_gradients = {}

    for name, grad in gradients.items():
        # Flatten gradient
        flat_grad = grad.flatten()

        # Calculate threshold (keep top k% values)
        k = int(flat_grad.numel() * (1 - sparsity))
        if k == 0:
            k = 1

        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(flat_grad.abs(), k)
        topk_signs = torch.sign(flat_grad[topk_indices])
        topk_values = topk_values * topk_signs

        sparse_gradients[name] = {
            'values': topk_values,
            'indices': topk_indices,
            'shape': grad.shape,
        }

    return sparse_gradients


def densify_gradients(sparse_gradients: dict) -> dict:
    """
    Convert sparse gradients back to dense format.

    Args:
        sparse_gradients: Dictionary of sparse gradient data

    Returns:
        Dictionary of dense gradients
    """
    dense_gradients = {}

    for name, sparse_data in sparse_gradients.items():
        shape = sparse_data['shape']
        values = sparse_data['values']
        indices = sparse_data['indices']

        # Create dense tensor
        dense_grad = torch.zeros(torch.prod(torch.tensor(shape)).item())
        dense_grad[indices] = values
        dense_grad = dense_grad.reshape(shape)

        dense_gradients[name] = dense_grad

    return dense_gradients


def compress_gradients(gradients: dict, method: str = 'quantize', **kwargs) -> dict:
    """
    Compress gradients using specified method.

    Args:
        gradients: Dictionary of gradient tensors
        method: Compression method ('quantize', 'sparsify', or 'none')
        **kwargs: Additional arguments for compression method

    Returns:
        Dictionary of compressed gradients
    """
    if method == 'quantize':
        num_bits = kwargs.get('num_bits', 8)
        return quantize_gradients(gradients, num_bits)
    elif method == 'sparsify':
        sparsity = kwargs.get('sparsity', 0.9)
        return sparsify_gradients(gradients, sparsity)
    elif method == 'none':
        return {'gradients': gradients, 'method': 'none'}
    else:
        raise ValueError(f"Unknown compression method: {method}")


def decompress_gradients(compressed_data: dict) -> dict:
    """
    Decompress gradients based on compression method.

    Args:
        compressed_data: Dictionary with compressed gradient data

    Returns:
        Dictionary of decompressed gradients
    """
    method = compressed_data.get('method', 'quantize')

    if method == 'quantize':
        return dequantize_gradients(compressed_data)
    elif method == 'sparsify':
        return densify_gradients(compressed_data['gradients'])
    elif method == 'none':
        return compressed_data['gradients']
    else:
        raise ValueError(f"Unknown compression method: {method}")


def estimate_compression_ratio(original: dict, compressed: dict) -> float:
    """
    Estimate compression ratio achieved.

    Args:
        original: Original gradients
        compressed: Compressed gradients

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    # Rough estimate based on number of elements
    original_size = sum(grad.numel() * 4 for grad in original.values())  # float32 = 4 bytes

    if 'params' in compressed:  # Quantized
        compressed_size = sum(grad.numel() for grad in compressed['gradients'].values())  # uint8 = 1 byte
        compressed_size += len(compressed['params']) * 16  # params overhead
    elif 'values' in list(compressed.values())[0]:  # Sparse
        compressed_size = sum(
            data['values'].numel() * 4 + data['indices'].numel() * 4
            for data in compressed.values()
        )
    else:
        compressed_size = original_size

    return original_size / compressed_size if compressed_size > 0 else 1.0
