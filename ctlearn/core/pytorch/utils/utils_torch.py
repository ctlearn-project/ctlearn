import torch

def cartesian_to_alt_az(directions):
    r = torch.sqrt(torch.sum(directions**2, dim=1))
    # Prevent division by zero
    safe_r = torch.where(r == 0, torch.tensor(1.0, device=r.device), r)

    # altitude_rad = torch.arcsin(safe_r)

    altitude_rad = torch.asin(directions[:, 2] / safe_r)
    azimuth_rad = torch.atan2(directions[:, 1], directions[:, 0])
    
    # Normalize azimuth to [0, 2π]
    # azimuth_rad = torch.where(azimuth_rad < 0, azimuth_rad + 2 * torch.pi, azimuth_rad)
    
    return torch.stack((altitude_rad, azimuth_rad), dim=1)

def alt_az_to_cartesian(altitude_rad, azimuth_rad, r=1):
    x = r * torch.cos(altitude_rad) * torch.cos(azimuth_rad)
    y = r * torch.cos(altitude_rad) * torch.sin(azimuth_rad)
    z = r * torch.sin(altitude_rad)
    return torch.stack((x, y, z), dim=1)  # Stack along new dimension to create vectors

def adjust_learning_rate( optimizer, lr):
    """Adjusts learning rate of all optimizer's parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compare_weights(model, initial_weights):
    for name, param in model.named_parameters():
        initial_weight = initial_weights[name]
        if not torch.equal(initial_weight, param.data):
            print(f"Weight changed: {name}")