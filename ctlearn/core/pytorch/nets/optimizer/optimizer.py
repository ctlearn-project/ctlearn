"""
Learning Rate Scheduling Functions Module

This module provides learning rate scheduling functions for training neural networks.
It includes implementations of various learning rate schedules commonly used in deep
learning, particularly focusing on one-cycle and cosine annealing schedules that have
been shown to improve training convergence and final model performance.

Functions:
    one_cycle: Generate a one-cycle learning rate schedule with sinusoidal ramp
    
References:
    - "A disciplined approach to neural network hyper-parameters" (Smith, 2018)
    - "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (Smith & Topin, 2017)
"""

import math

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """
    Generate a one-cycle learning rate schedule using a sinusoidal ramp.
    
    This function creates a lambda function that implements a smooth, cosine-based
    transition from an initial learning rate (y1) to a maximum learning rate (y2)
    over a specified number of steps. The schedule follows a half-cosine curve,
    providing a gradual warm-up and smooth transition.
    
    The one-cycle policy has been shown to enable faster training and better
    generalization by allowing the use of higher learning rates during training
    while maintaining stability through the smooth schedule.
    
    Mathematical Formula:
        lr(x) = ((1 - cos(x * π / steps)) / 2) * (y2 - y1) + y1
        
        where:
        - x: current step (0 to steps)
        - lr(x): learning rate at step x
        - The cosine creates a smooth S-shaped curve from y1 to y2
    
    Args:
        y1 (float, optional): Initial learning rate (start value). Defaults to 0.0
            Typically set to a small value or 0 for warm-up from zero
        y2 (float, optional): Maximum learning rate (end value). Defaults to 1.0
            This is the peak learning rate to reach during training
            Should be set based on learning rate range tests
        steps (int, optional): Total number of steps for the schedule. Defaults to 100
            This determines how quickly the learning rate increases
            For epoch-based: steps = num_epochs
            For iteration-based: steps = num_epochs * batches_per_epoch
    
    Returns:
        function: A lambda function that takes step index x and returns the 
            corresponding learning rate. The function signature is:
            lambda x: float
            
    Schedule Characteristics:
        - At x=0: Returns y1 (starting learning rate)
        - At x=steps/2: Returns (y1+y2)/2 (midpoint)
        - At x=steps: Returns y2 (maximum learning rate)
        - Smooth acceleration: No sudden jumps in learning rate
        - Cosine-based: Provides gentle start and smooth transition
    
    Usage Patterns:
        1. One-Cycle Policy (Smith, 2018):
           - Phase 1: Ramp up from low LR to high LR (this function)
           - Phase 2: Ramp down from high LR to very low LR (inverse)
           
        2. Warm-up Only:
           - Use this function alone for gradual LR warm-up
           - Helps stabilize training at the beginning
    
    Example:
        >>> # Create a schedule from 0.0 to 0.1 over 1000 steps
        >>> schedule = one_cycle(y1=0.0, y2=0.1, steps=1000)
        >>> 
        >>> # Get learning rate at step 0 (start)
        >>> lr_start = schedule(0)
        >>> print(f"LR at step 0: {lr_start:.6f}")  # 0.000000
        >>> 
        >>> # Get learning rate at step 500 (halfway)
        >>> lr_mid = schedule(500)
        >>> print(f"LR at step 500: {lr_mid:.6f}")  # ~0.050000
        >>> 
        >>> # Get learning rate at step 1000 (end)
        >>> lr_end = schedule(1000)
        >>> print(f"LR at step 1000: {lr_end:.6f}")  # 0.100000
        
        >>> # Use with PyTorch LambdaLR scheduler
        >>> import torch.optim as optim
        >>> optimizer = optim.SGD(model.parameters(), lr=1.0)
        >>> lambda_func = one_cycle(y1=0.0, y2=1.0, steps=100)
        >>> scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
        >>> 
        >>> # During training
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     scheduler.step()  # Updates LR according to schedule
        
        >>> # Complete one-cycle schedule (warm-up + cool-down)
        >>> warmup = one_cycle(0.0, 1.0, steps=50)
        >>> cooldown = one_cycle(1.0, 0.01, steps=50)
        >>> # Use warmup for first 50 epochs, cooldown for last 50
    
    Notes:
        - The returned lambda is stateless; it only depends on the input step
        - Can be used with PyTorch's LambdaLR scheduler for automatic LR updates
        - Works with both epoch-based and iteration-based schedules
        - The cosine curve provides smoother transitions than linear schedules
        - Avoids sudden LR changes that can destabilize training
        
    Common Configurations:
        Warm-up from zero:
            >>> schedule = one_cycle(0.0, 0.001, steps=10)  # 10-epoch warm-up
            
        One-cycle for 100 epochs:
            >>> up = one_cycle(0.0, 0.1, steps=70)      # Ramp up: epochs 0-70
            >>> down = one_cycle(0.1, 0.0001, steps=30) # Ramp down: epochs 70-100
            
        Fine-tuning with low LR:
            >>> schedule = one_cycle(0.00001, 0.0001, steps=20)  # Gentle increase
    
    See Also:
        torch.optim.lr_scheduler.LambdaLR: PyTorch scheduler using lambda functions
        torch.optim.lr_scheduler.OneCycleLR: Built-in one-cycle implementation
        torch.optim.lr_scheduler.CosineAnnealingLR: Cosine annealing schedule
    """
    # Return lambda function that computes LR for any step x
    # The cosine function creates a smooth S-curve from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
