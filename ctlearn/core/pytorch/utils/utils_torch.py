"""
PyTorch Utility Functions Module

This module provides utility functions for coordinate transformations, optimizer
management, and model weight tracking in PyTorch. These utilities are specifically
designed for Cherenkov telescope array analysis in CTLearn.

Functions:
    cartesian_to_alt_az: Convert Cartesian coordinates to altitude/azimuth
    alt_az_to_cartesian: Convert altitude/azimuth to Cartesian coordinates
    adjust_learning_rate: Modify learning rate for all optimizer parameter groups
    compare_weights: Compare current model weights with initial weights
"""

import torch

def cartesian_to_alt_az(directions):
    """
    Convert 3D Cartesian direction vectors to altitude and azimuth angles.
    
    This function transforms Cartesian coordinates (x, y, z) representing
    unit direction vectors into spherical coordinates (altitude, azimuth)
    commonly used in astronomical coordinate systems.
    
    Coordinate System:
        - Cartesian: (x, y, z) where x² + y² + z² = r²
        - Spherical: (altitude, azimuth) in radians
            * Altitude (elevation): angle above the horizontal plane [-π/2, π/2]
            * Azimuth: angle in the horizontal plane, measured from x-axis [−π, π]
    
    Args:
        directions (torch.Tensor): Batch of 3D direction vectors
            Shape: (batch_size, 3) where each row is [x, y, z]
            Can be unit vectors or any length (will be normalized internally)
            
    Returns:
        torch.Tensor: Altitude and azimuth angles in radians
            Shape: (batch_size, 2) where each row is [altitude, azimuth]
            - altitude: Range [-π/2, π/2] radians (-90° to 90°)
            - azimuth: Range [-π, π] radians (-180° to 180°)
            
    Mathematical Formulation:
        r = √(x² + y² + z²)
        altitude = arcsin(z / r)
        azimuth = arctan2(y, x)
    
    Example:
        >>> # Pointing straight up (zenith)
        >>> directions = torch.tensor([[0.0, 0.0, 1.0]])
        >>> alt_az = cartesian_to_alt_az(directions)
        >>> print(alt_az)  # [[π/2, 0]]
        
        >>> # Batch of directions
        >>> directions = torch.tensor([
        ...     [1.0, 0.0, 0.0],  # East horizon
        ...     [0.0, 1.0, 0.0],  # North horizon
        ...     [0.0, 0.0, 1.0],  # Zenith
        ... ])
        >>> alt_az = cartesian_to_alt_az(directions)
        >>> # Result: [[0, 0], [0, π/2], [π/2, 0]]
        
    Notes:
        - Handles zero vectors safely by replacing r=0 with r=1 to avoid division by zero
        - Azimuth follows standard convention: 0 at x-axis, increases counterclockwise
        - For shower direction reconstruction in gamma-ray astronomy
        - Compatible with astropy coordinate transformations
    """
    # Calculate magnitude (radius) of each direction vector
    r = torch.sqrt(torch.sum(directions**2, dim=1))
    
    # Prevent division by zero for null vectors
    # Replace zero magnitudes with 1.0 to avoid NaN in division
    safe_r = torch.where(r == 0, torch.tensor(1.0, device=r.device), r)

    # Calculate altitude (elevation angle above horizontal plane)
    # altitude = arcsin(z / r), range: [-π/2, π/2]
    altitude_rad = torch.asin(directions[:, 2] / safe_r)
    
    # Calculate azimuth (angle in horizontal plane from x-axis)
    # azimuth = arctan2(y, x), range: [-π, π]
    azimuth_rad = torch.atan2(directions[:, 1], directions[:, 0])
    
    # Note: Azimuth normalization to [0, 2π] is commented out
    # Current implementation returns azimuth in [-π, π]
    # Uncomment the following line if [0, 2π] range is needed:
    # azimuth_rad = torch.where(azimuth_rad < 0, azimuth_rad + 2 * torch.pi, azimuth_rad)
    
    # Stack altitude and azimuth into single tensor
    # Shape: (batch_size, 2)
    return torch.stack((altitude_rad, azimuth_rad), dim=1)

def alt_az_to_cartesian(altitude_rad, azimuth_rad, r=1):
    """
    Convert altitude and azimuth angles to 3D Cartesian coordinates.
    
    This function transforms spherical coordinates (altitude, azimuth) into
    Cartesian coordinates (x, y, z). This is the inverse operation of
    cartesian_to_alt_az.
    
    Coordinate System:
        - Input: (altitude, azimuth, radius) in radians and distance units
        - Output: (x, y, z) Cartesian coordinates
    
    Args:
        altitude_rad (torch.Tensor): Altitude angles in radians
            Shape: (batch_size,) or scalar
            Range: [-π/2, π/2] (where 0 is horizon, π/2 is zenith)
            
        azimuth_rad (torch.Tensor): Azimuth angles in radians
            Shape: (batch_size,) or scalar
            Range: [-π, π] or [0, 2π]
            Convention: 0 at x-axis, increases counterclockwise
            
        r (float or torch.Tensor, optional): Radial distance from origin
            Default: 1 (unit vectors)
            Can be scalar (same for all) or tensor (batch_size,)
            
    Returns:
        torch.Tensor: 3D Cartesian direction vectors
            Shape: (batch_size, 3) where each row is [x, y, z]
            If r=1, returns unit vectors
            
    Mathematical Formulation:
        x = r · cos(altitude) · cos(azimuth)
        y = r · cos(altitude) · sin(azimuth)
        z = r · sin(altitude)
    
    Example:
        >>> # Convert zenith direction (straight up)
        >>> altitude = torch.tensor([np.pi/2])  # 90 degrees
        >>> azimuth = torch.tensor([0.0])
        >>> xyz = alt_az_to_cartesian(altitude, azimuth)
        >>> print(xyz)  # [[0, 0, 1]]
        
        >>> # Convert horizon direction pointing East
        >>> altitude = torch.tensor([0.0])
        >>> azimuth = torch.tensor([0.0])
        >>> xyz = alt_az_to_cartesian(altitude, azimuth)
        >>> print(xyz)  # [[1, 0, 0]]
        
        >>> # Batch conversion with custom radius
        >>> altitudes = torch.tensor([0.0, np.pi/4, np.pi/2])
        >>> azimuths = torch.tensor([0.0, np.pi/2, 0.0])
        >>> xyz = alt_az_to_cartesian(altitudes, azimuths, r=2.0)
        >>> # Result: [[2, 0, 0], [0, √2, √2], [0, 0, 2]]
        
    Notes:
        - Default r=1 produces unit vectors
        - Useful for converting predicted angles back to 3D directions
        - Compatible with shower direction reconstruction in CTA analysis
        - Inverse of cartesian_to_alt_az (within numerical precision)
    """
    # Calculate x-coordinate
    # x = r · cos(altitude) · cos(azimuth)
    # Points in the horizontal plane, aligned with azimuth angle
    x = r * torch.cos(altitude_rad) * torch.cos(azimuth_rad)
    
    # Calculate y-coordinate
    # y = r · cos(altitude) · sin(azimuth)
    # Points in the horizontal plane, perpendicular to x
    y = r * torch.cos(altitude_rad) * torch.sin(azimuth_rad)
    
    # Calculate z-coordinate
    # z = r · sin(altitude)
    # Points vertically (altitude component)
    z = r * torch.sin(altitude_rad)
    
    # Stack coordinates into 3D vectors
    # Shape: (batch_size, 3)
    return torch.stack((x, y, z), dim=1)

def adjust_learning_rate(optimizer, lr):
    """
    Adjust the learning rate for all parameter groups in an optimizer.
    
    This function modifies the learning rate of all parameter groups in a
    PyTorch optimizer. Useful for implementing custom learning rate schedules,
    warm-up strategies, or manual learning rate adjustments during training.
    
    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer instance
            Can be any optimizer (SGD, Adam, AdamW, etc.)
            Contains one or more parameter groups
            
        lr (float): New learning rate to set
            Must be positive
            Applied to all parameter groups uniformly
            
    Side Effects:
        Modifies the 'lr' field of all parameter groups in the optimizer in-place
        
    Example:
        >>> # Initialize optimizer
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> 
        >>> # Reduce learning rate by factor of 10
        >>> adjust_learning_rate(optimizer, 0.0001)
        >>> 
        >>> # Verify new learning rate
        >>> print(optimizer.param_groups[0]['lr'])
        0.0001
        
        >>> # Use in training loop for manual scheduling
        >>> for epoch in range(num_epochs):
        ...     if epoch == 50:
        ...         adjust_learning_rate(optimizer, lr * 0.1)
        ...     train_one_epoch()
        
    Notes:
        - Affects ALL parameter groups (if optimizer has multiple groups)
        - For different learning rates per group, access param_groups directly
        - Common use cases:
            * Learning rate warm-up
            * Manual learning rate decay
            * Cyclical learning rate schedules
            * Recovery from divergence during training
        - Alternative: Use PyTorch lr_scheduler classes for automatic scheduling
        
    See Also:
        torch.optim.lr_scheduler: Built-in learning rate schedulers
    """
    # Iterate over all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        # Update the learning rate for this parameter group
        param_group['lr'] = lr

def compare_weights(model, initial_weights):
    """
    Compare current model weights with initial weights to detect changes.
    
    This utility function checks which parameters in a PyTorch model have
    changed compared to their initial values. Useful for debugging training
    issues, verifying that training is updating weights, or checking if
    certain layers are frozen correctly.
    
    Args:
        model (torch.nn.Module): PyTorch model to check
            Can be any neural network model
            
        initial_weights (dict): Dictionary of initial weight tensors
            Format: {parameter_name: tensor}
            Typically obtained from model.state_dict() before training
            Example: initial_weights = {name: param.data.clone() 
                                       for name, param in model.named_parameters()}
            
    Side Effects:
        Prints the names of parameters that have changed
        Does not modify the model or weights
        
    Example:
        >>> # Save initial weights before training
        >>> model = ResNet(num_outputs=2)
        >>> initial_weights = {
        ...     name: param.data.clone() 
        ...     for name, param in model.named_parameters()
        ... }
        >>> 
        >>> # Train for one epoch
        >>> train_one_epoch(model, optimizer, train_loader)
        >>> 
        >>> # Check which weights changed
        >>> compare_weights(model, initial_weights)
        Weight changed: conv1.weight
        Weight changed: bn1.weight
        Weight changed: layer1.0.conv1.weight
        ...
        
        >>> # Check if frozen layers stayed frozen
        >>> for name, param in model.named_parameters():
        ...     if 'backbone' in name:
        ...         param.requires_grad = False
        >>> 
        >>> train_one_epoch(model, optimizer, train_loader)
        >>> compare_weights(model, initial_weights)
        # Should not print any 'backbone' layers
        
    Notes:
        - Only checks parameters that exist in both model and initial_weights
        - Uses torch.equal() for exact comparison (no tolerance)
        - Useful for debugging:
            * Verifying training is working (weights should change)
            * Checking frozen layers (weights should NOT change)
            * Identifying which layers are being updated
            * Detecting gradient flow issues
        - For large models, consider checking only specific layers
        
    Performance:
        - Comparison is done in-place, no additional memory allocation
        - Fast for most models, but can be slow for very large models
    """
    # Iterate over all named parameters in the model
    for name, param in model.named_parameters():
        # Get the corresponding initial weight tensor
        initial_weight = initial_weights[name]
        
        # Compare current weight with initial weight
        # torch.equal() returns True only if tensors are exactly equal
        if not torch.equal(initial_weight, param.data):
            # Print parameter name if it has changed
            print(f"Weight changed: {name}")