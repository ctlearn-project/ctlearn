import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class evidential_regression_loss(nn.Module):
    def __init__(self, lamb=1.0, reduction='mean'):
        super(evidential_regression_loss, self).__init__()
        self.reduction = reduction
        self.lamb = lamb
    def nig_nll(self, mu, v, alpha, beta, y):
        """Computes the Negative Log-Likelihood for Normal Inverse Gamma."""
        two_beta_lambda = 2 * beta * (1 + v)
        t1 = 0.5 * (torch.pi / v).log()
        t2 = alpha * two_beta_lambda.log()
        t3 = (alpha + 0.5) * (v * (y - mu) ** 2 + two_beta_lambda).log()
        t4 = alpha.lgamma()
        t5 = (alpha + 0.5).lgamma()
        nll = t1 - t2 + t3 + t4 - t5
        return nll

    def nig_reg(self, mu, v, alpha, _beta, y):
        """Computes the Normal Inverse Gamma regularization."""
        reg = (y - mu).abs() * (2 * v + alpha)
        if self.reduction=="mean":
            error = reg.mean() 
        elif self.reduction=="sum":
            error = reg.sum() 
        elif self.reduction == None or self.reduction =='None':
            error = reg

        else: 
            raise RuntimeError("Reduction not supported: Use sum or mean")
        
        return error
    def set_lambda(self, lamb):
        self.lamb =lamb 
        
    def forward(self, dist_params, y):
        """Computes the evidential regression loss."""
        if len(y)>1:
            mu, v, alpha, beta = (d.squeeze() for d in dist_params)
        else: 
            mu, v, alpha, beta = (d for d in dist_params)
        # mu, v, alpha, beta = (d for d in dist_params)

        nig_reg_error = self.nig_reg( mu, v, alpha, beta, y)
        nig_nll_error = self.nig_nll( mu, v, alpha, beta, y)

        if self.reduction=="mean":
            nig_nll_error = nig_nll_error.mean() 
        elif self.reduction=="sum":
            nig_nll_error = nig_nll_error.sum() 
        elif self.reduction == None or self.reduction =='None':
            nig_nll_error = nig_nll_error
        else: 
            raise RuntimeError("Reduction not supported: Use sum or mean")

        return nig_nll_error + self.lamb *nig_reg_error

def cosine_direction_loss(pred_x, pred_y, true_x, true_y,reduction="mean"):
    pred_vec = F.normalize(torch.stack([pred_x, pred_y], dim=1), dim=1)
    true_vec = F.normalize(torch.stack([true_x, true_y], dim=1), dim=1)
    if reduction=="mean":
        return 1 - torch.sum(pred_vec * true_vec, dim=1).mean()
    elif reduction=="sum":
        return 1 - torch.sum(pred_vec * true_vec, dim=1).sum()
    elif reduction=="none":
        return 1 - torch.sum(pred_vec * true_vec, dim=1) 
    else: 
        raise RuntimeError("Reduction not supported: Use sum , mean or none")

def AngularDistance(alt1_rad, alt2_rad, az1_rad, az2_rad,reduction = None):
    """
    Calculate the angular distance between points given in batches.
    
    Parameters:
    - alt1_rad, az1_rad: Tensors of the altitudes and azimuths in radians for the first set of points.
    - alt2_rad, az2_rad: Tensors of the altitudes and azimuths in radians for the second set of points.
    
    Returns:
    - Tensor of angular distances in radians for each pair of points.
    """
    
    # Compute the cosine of the angular distance using batch-wise operations
    cosdelta = torch.cos(alt1_rad) * torch.cos(alt2_rad) * torch.cos(az1_rad - az2_rad) + \
               torch.sin(alt1_rad) * torch.sin(alt2_rad)
    
    # Clamp the cosdelta values to ensure they are within the valid range for arccos
    # cosdelta = torch.clamp(cosdelta, -1.0, 1.0)
    cosdelta = torch.clamp(cosdelta, -1.0 + 1e-7, 1.0 - 1e-7)
    # Calculate the angular distance in radians
    ang_dist_rad = torch.acos(cosdelta)
    ang_dist_rad[cosdelta == 1.0] = 0.0  # acos(1) = 0
    ang_dist_rad[cosdelta == -1.0] = torch.pi  # acos(-1) = pi    
    ang_dist_deg= torch.rad2deg(ang_dist_rad)

    if reduction =="sum":
        return ang_dist_rad.sum(),ang_dist_deg.sum()
    elif reduction=="mean":
        return ang_dist_rad.mean(),ang_dist_deg.mean()
    elif reduction == None or reduction =='None':
        return ang_dist_rad, ang_dist_deg
    else: 
        raise RuntimeError("Reduction not supported: Use sum, mean or None")
    
def AngularError(vec1, vec2,reduction='mean'):
    # Ensure the vectors are tensors
    # vec1 = vec1.clone().detach().float()
    # vec2 = vec2.clone().detach().float()


    # Compute the dot product for each pair of vectors in the batch
    dot_product = torch.sum(vec1 * vec2, dim=1)

    # Compute the magnitudes (norms) of the vectors for each vector in the batch
    norm_vec1 = torch.norm(vec1, dim=1)
    norm_vec2 = torch.norm(vec2, dim=1)

    # Compute the cosine of the angle for each pair of vectors in the batch
    cos_theta = dot_product / (norm_vec1 * norm_vec2)

    # Clip the cosine values to the range [-1, 1] to avoid numerical issues with arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute the angle in radians for each pair of vectors in the batch
    angle_rad = torch.acos(cos_theta)

    # Optionally, convert the angles from radians to degrees
    angle_deg = torch.rad2deg(angle_rad)

    if reduction =="sum":
        return angle_rad.sum(), angle_deg.sum()
    elif reduction=="mean":
        return angle_rad.mean(), angle_deg.mean()
    elif reduction == None or reduction =='None':
        return angle_rad, angle_deg
    else: 
        raise RuntimeError("Reduction not supported: Use sum, mean or None")


class VectorLoss(nn.Module):
    def __init__(self,alpha=0.001,reduction='mean'):
        super(VectorLoss, self).__init__()
        self.alpha = alpha
        self.reduction=reduction
    def forward(self, output, target):
        # Calculate angles of output and target using atan2
        angles_output = torch.atan2(output[:, 1], output[:, 0])
        angles_target = torch.atan2(target[:, 1], target[:, 0])

        # Compute the difference in angles
        angle_diff = torch.abs(angles_output - angles_target)

        # Normalize angle differences to be within [0, pi]
        angle_diff = torch.remainder(angle_diff + torch.pi, 2 * torch.pi) - torch.pi
        angle_diff = torch.abs(angle_diff)  # Ensure all differences are positive

        if self.reduction =="sum":
            return angle_diff.sum()
        elif self.reduction=="mean":
            return angle_diff.mean()
        else: 
            raise RuntimeError("Reduction not supported: Use sum or mean")
        return self.alpha * angle_diff.mean()
def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def generate_hot_ones(device, cn, cp, outputs, targets):

    t = torch.full_like(outputs, cn, device=device)
    n = outputs.shape[0]
    t[range(n), targets] = cp

    return t

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction 
        
    def set_alpha(self,alpha):
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilidad inversa del error
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
     
# class FocalLoss(nn.Module):
#     def __init__(self, device, alpha=0.25, gamma=2.0, label_smoothing=0.0):
#         super(FocalLoss, self).__init__()
#         self.device = device
#         self.alpha = alpha
#         self.gamma = gamma
#         self.label_smoothing = label_smoothing
#         self.cp, self.cn = smooth_BCE(eps=self.label_smoothing)
#         self.BCE = BCELogitsLoss(device)

#     def forward(self, outputs, targets):

#         # t = generate_hot_ones(self.device, self.cn, self.cp, outputs, targets)
#         # Supone inputs son las logits antes de sigmoid
#         # BCE_loss = F.binary_cross_entropy_with_logits(outputs, t, reduction="none")
#         BCE_loss = self.BCE(outputs,targets)
#         pt = torch.exp(-BCE_loss)  # pt es la probabilidad de clasificar correctamente
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()


class BCELogitsLoss(nn.Module):
    def __init__(self, device, cls_pw=1.0, label_smoothing=0.0):
        super().__init__()
        self.device = device
        self.label_smoothing = label_smoothing
        self.BCE = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cls_pw, device=self.device)
        )
        self.cp, self.cn = smooth_BCE(eps=self.label_smoothing)

    def forward(self, outputs, targets):

        # Generate hot ones targets
        t = generate_hot_ones(self.device, self.cn, self.cp, outputs, targets)

        bce_loss = self.BCE(outputs, t)

        return bce_loss




class EvidClassification():
    def __init__(self,class_weights=None):
        self.class_weights= class_weights

    def dirichlet_reg(self, alpha, y):
        # dirichlet parameters after removal of non-misleading evidence (from the label)
        alpha = y + (1 - y) * alpha

        # uniform dirichlet distribution
        beta = torch.ones_like(alpha)

        sum_alpha = alpha.sum(-1)
        sum_beta = beta.sum(-1)

        t1 = sum_alpha.lgamma() - sum_beta.lgamma()
        t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
        t3 = alpha - beta
        t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

        kl = t1 - t2 + (t3 * t4).sum(-1)
        return kl.sum()

    def dirichlet_mse(self, alpha, y, ):
        sum_alpha = alpha.sum(-1, keepdims=True)
        p = alpha / sum_alpha
        t1 = (y - p).pow(2)
        t2 = ((p * (1 - p)) / (sum_alpha + 1))

        if  self.class_weights is not None:
            t1 = t1 *  self.class_weights.unsqueeze(0)
            t2 = t2 *  self.class_weights.unsqueeze(0)

        mse = t1 + t2
        return mse.sum()

    def loss(self, alpha, y, lamb=1.0 ):
        num_classes = alpha.shape[-1]
        y = F.one_hot(y, num_classes)
        return self.dirichlet_mse(alpha, y) + lamb * self.dirichlet_reg(alpha, y)

# def evidential_classification(alpha, y, lamb=1.0):
#     num_classes = alpha.shape[-1]
#     y = F.one_hot(y, num_classes)
#     return dirichlet_mse(alpha, y) + lamb * dirichlet_reg(alpha, y)

# def evidential_classification(alpha, y, weights, lamb=1.0):
#     num_classes = alpha.shape[-1]
#     y = F.one_hot(y, num_classes)
#     mse_loss = dirichlet_mse(alpha, y)
#     reg_loss = dirichlet_reg(alpha, y)
#     weighted_loss = weights[0] * mse_loss + weights[1] * reg_loss
#     return weighted_loss + lamb * reg_loss

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):

#         t = generate_hot_ones(self.device, self.cn, self.cp, targets, targets)
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at * (1-pt)**self.gamma * BCE_loss

#         if self.reduction == 'mean':
#             return F_loss.mean()
#         elif self.reduction == 'sum':
#             return F_loss.sum()
#         else:
#             return F_loss
        

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.float32)
#         at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#         pt = torch.exp(-BCE_loss)
#         F_loss = at * (1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()        
