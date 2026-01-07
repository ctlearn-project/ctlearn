"""
Neural Network Activation Functions Module

This module provides custom activation functions and utilities for neural networks
in CTLearn. It includes both standard PyTorch activations and custom implementations
optimized for Cherenkov telescope data analysis.

Activation functions introduce non-linearity into neural networks, allowing them
to learn complex patterns beyond simple linear transformations. The choice of
activation function can significantly impact model performance, training stability,
and convergence speed.

Common Uses:
    - ReLU and variants: Standard choice for hidden layers in CNNs
    - Sigmoid: Binary classification output layers
    - Tanh: Alternative to sigmoid with zero-centered output
    - Softmax: Multi-class classification output layers
    - Custom activations: Task-specific optimizations

Imports:
    torch: PyTorch tensor operations
    torch.nn: Neural network modules and activation functions

Example:
    >>> import torch.nn as nn
    >>> # Using standard PyTorch activations
    >>> activation = nn.ReLU()
    >>> x = torch.tensor([-1.0, 0.0, 1.0])
    >>> output = activation(x)  # [0.0, 0.0, 1.0]
    
    >>> # In a neural network layer
    >>> layer = nn.Sequential(
    ...     nn.Conv2d(1, 64, kernel_size=3),
    ...     nn.ReLU(),
    ...     nn.BatchNorm2d(64)
    ... )

Notes:
    - This module is a placeholder for future custom activation functions
    - Standard PyTorch activations (nn.ReLU, nn.Sigmoid, etc.) are used throughout CTLearn
    - Custom activations can be added here when needed for specific tasks
    
Future Extensions:
    - Swish/SiLU activation: x * sigmoid(x), shown to improve performance in some tasks
    - GELU: Gaussian Error Linear Unit, used in transformers
    - Mish: Self-regularized non-monotonic activation
    - Parametric activations: PReLU, ELU with learnable parameters
"""

import torch
import torch.nn as nn

# This module currently serves as a placeholder for custom activation functions.
# Standard PyTorch activation functions are used directly from torch.nn:
#
# Common Activations Available:
# - nn.ReLU(): Rectified Linear Unit, f(x) = max(0, x)
# - nn.LeakyReLU(negative_slope): Leaky ReLU, f(x) = max(negative_slope*x, x)
# - nn.ELU(alpha): Exponential Linear Unit
# - nn.GELU(): Gaussian Error Linear Unit
# - nn.Sigmoid(): Sigmoid function, f(x) = 1 / (1 + exp(-x))
# - nn.Tanh(): Hyperbolic tangent, f(x) = tanh(x)
# - nn.Softmax(dim): Softmax for multi-class classification
# - nn.LogSoftmax(dim): Log of softmax, numerically stable
#
# Usage Example:
# from torch.nn import ReLU, Sigmoid
# activation = ReLU()
# output = activation(input_tensor)

