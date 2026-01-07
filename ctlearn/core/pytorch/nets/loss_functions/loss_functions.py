"""
Loss Functions Module for Deep Learning

This module provides custom loss functions for various deep learning tasks in CTLearn,
with a focus on uncertainty quantification through evidential learning and specialized
losses for astronomical data analysis.

Loss Functions:
    evidential_regression_loss: Normal Inverse Gamma loss for regression with uncertainty
    cosine_direction_loss: Cosine similarity loss for direction reconstruction
    AngularDistance: Angular distance metric for celestial coordinates
    AngularError: Angular error between vector predictions
    VectorLoss: Angle-based loss for 2D vectors
    FocalLoss: Focal loss for imbalanced classification
    BCELogitsLoss: Binary cross-entropy with label smoothing
    EvidClassification: Evidential classification with Dirichlet distribution

Utility Functions:
    smooth_BCE: Generate smoothed BCE targets
    generate_hot_ones: Create one-hot encoded targets with smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class evidential_regression_loss(nn.Module):
    """
    Evidential regression loss using Normal Inverse Gamma (NIG) distribution.
    
    This loss function implements evidential deep learning for regression tasks,
    providing both point predictions and uncertainty estimates. It combines
    a negative log-likelihood term with a regularization term to prevent
    overconfident predictions.
    
    Mathematical Background:
        The NIG distribution is parameterized by (μ, v, α, β):
        - μ: Predicted mean
        - v: Virtual observation count (inverse epistemic uncertainty)
        - α: Shape parameter
        - β: Scale parameter (related to aleatoric uncertainty)
        
        Total Loss = NLL + λ * Regularization
        
    Attributes:
        lamb (float): Weight for the regularization term
        reduction (str): Reduction method ('mean', 'sum', or None)
    """
    
    def __init__(self, lamb=1.0, reduction='mean'):
        """
        Initialize the evidential regression loss.
        
        Args:
            lamb (float, optional): Regularization weight. Defaults to 1.0
                Higher values encourage more conservative uncertainty estimates
            reduction (str, optional): Reduction method. Defaults to 'mean'
                Options: 'mean', 'sum', None
        """
        super(evidential_regression_loss, self).__init__()
        self.reduction = reduction
        self.lamb = lamb
        
    def nig_nll(self, mu, v, alpha, beta, y):
        """
        Compute the Negative Log-Likelihood for Normal Inverse Gamma distribution.
        
        This method calculates the NLL which measures how well the predicted
        NIG distribution matches the observed target values.
        
        Args:
            mu (torch.Tensor): Predicted mean values
            v (torch.Tensor): Virtual observation counts
            alpha (torch.Tensor): Shape parameters
            beta (torch.Tensor): Scale parameters
            y (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Negative log-likelihood values
        """
        two_beta_lambda = 2 * beta * (1 + v)
        t1 = 0.5 * (torch.pi / v).log()
        t2 = alpha * two_beta_lambda.log()
        t3 = (alpha + 0.5) * (v * (y - mu) ** 2 + two_beta_lambda).log()
        t4 = alpha.lgamma()
        t5 = (alpha + 0.5).lgamma()
        nll = t1 - t2 + t3 + t4 - t5
        return nll

    def nig_reg(self, mu, v, alpha, _beta, y):
        """
        Compute the Normal Inverse Gamma regularization term.
        
        This regularization penalizes large prediction errors when the model
        is highly confident (high evidence), encouraging the model to match
        its uncertainty to its actual performance.
        
        Args:
            mu (torch.Tensor): Predicted mean values
            v (torch.Tensor): Virtual observation counts
            alpha (torch.Tensor): Shape parameters
            _beta (torch.Tensor): Scale parameters (not used)
            y (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Regularization values
            
        Raises:
            RuntimeError: If reduction method is not 'sum', 'mean', or None
        """
        # Regularization based on prediction error weighted by evidence
        reg = (y - mu).abs() * (2 * v + alpha)
        
        if self.reduction == "mean":
            error = reg.mean() 
        elif self.reduction == "sum":
            error = reg.sum() 
        elif self.reduction == None or self.reduction == 'None':
            error = reg
        else: 
            raise RuntimeError("Reduction not supported: Use sum or mean")
        
        return error
    
    def set_lambda(self, lamb):
        """
        Update the regularization weight.
        
        Args:
            lamb (float): New regularization weight
        """
        self.lamb = lamb 
        
    def forward(self, dist_params, y):
        """
        Compute the evidential regression loss.
        
        Args:
            dist_params (tuple): Tuple of (mu, v, alpha, beta) parameters
            y (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Combined NLL and regularization loss
        """
        # Unpack distribution parameters
        if len(y) > 1:
            mu, v, alpha, beta = (d.squeeze() for d in dist_params)
        else: 
            mu, v, alpha, beta = (d for d in dist_params)

        # Compute regularization term
        nig_reg_error = self.nig_reg(mu, v, alpha, beta, y)
        
        # Compute negative log-likelihood
        nig_nll_error = self.nig_nll(mu, v, alpha, beta, y)

        # Apply reduction to NLL
        if self.reduction == "mean":
            nig_nll_error = nig_nll_error.mean() 
        elif self.reduction == "sum":
            nig_nll_error = nig_nll_error.sum() 
        elif self.reduction == None or self.reduction == 'None':
            nig_nll_error = nig_nll_error
        else: 
            raise RuntimeError("Reduction not supported: Use sum or mean")

        # Combine NLL and weighted regularization
        return nig_nll_error + self.lamb * nig_reg_error

def cosine_direction_loss(pred_x, pred_y, true_x, true_y, reduction="mean"):
    """
    Compute cosine similarity loss for 2D direction vectors.
    
    This loss function measures the angular difference between predicted and
    true direction vectors using cosine similarity. It's particularly useful
    for shower direction reconstruction where we care about the angle rather
    than the magnitude.
    
    Mathematical Formula:
        loss = 1 - cos(θ) = 1 - (pred · true) / (|pred| |true|)
        
    Args:
        pred_x (torch.Tensor): Predicted x-components
        pred_y (torch.Tensor): Predicted y-components
        true_x (torch.Tensor): True x-components
        true_y (torch.Tensor): True y-components
        reduction (str, optional): Reduction method. Defaults to "mean"
            Options: 'mean', 'sum', 'none'
            
    Returns:
        torch.Tensor: Cosine direction loss
        
    Raises:
        RuntimeError: If reduction method is not supported
        
    Note:
        Both predicted and true vectors are normalized before computing
        the dot product, making the loss independent of vector magnitude.
    """
    # Normalize prediction vectors to unit length
    pred_vec = F.normalize(torch.stack([pred_x, pred_y], dim=1), dim=1)
    # Normalize true vectors to unit length
    true_vec = F.normalize(torch.stack([true_x, true_y], dim=1), dim=1)
    
    # Compute 1 - cosine similarity (0 for perfect alignment, 2 for opposite directions)
    if reduction == "mean":
        return 1 - torch.sum(pred_vec * true_vec, dim=1).mean()
    elif reduction == "sum":
        return 1 - torch.sum(pred_vec * true_vec, dim=1).sum()
    elif reduction == "none":
        return 1 - torch.sum(pred_vec * true_vec, dim=1) 
    else: 
        raise RuntimeError("Reduction not supported: Use sum, mean or none")

def AngularDistance(alt1_rad, alt2_rad, az1_rad, az2_rad, reduction=None):
    """
    Calculate the angular distance between celestial coordinates.
    
    This function computes the great circle distance between two points on
    the celestial sphere using the spherical law of cosines. Used for
    evaluating direction reconstruction accuracy in astronomy.
    
    Mathematical Formula:
        cos(Δθ) = cos(alt1)cos(alt2)cos(az1-az2) + sin(alt1)sin(alt2)
        Δθ = arccos(cos(Δθ))
    
    Args:
        alt1_rad (torch.Tensor): Altitude of first points in radians
        alt2_rad (torch.Tensor): Altitude of second points in radians
        az1_rad (torch.Tensor): Azimuth of first points in radians
        az2_rad (torch.Tensor): Azimuth of second points in radians
        reduction (str, optional): Reduction method. Defaults to None
            Options: 'sum', 'mean', None
            
    Returns:
        tuple: (angular_distance_rad, angular_distance_deg)
            - angular_distance_rad: Angular distances in radians
            - angular_distance_deg: Angular distances in degrees
            
    Raises:
        RuntimeError: If reduction method is not supported
        
    Note:
        - Handles numerical edge cases (cosdelta = ±1) explicitly
        - Clamps cosdelta to prevent arccos domain errors
        - Returns both radians and degrees for convenience
    """
    # Compute cosine of angular distance using spherical law of cosines
    cosdelta = torch.cos(alt1_rad) * torch.cos(alt2_rad) * torch.cos(az1_rad - az2_rad) + \
               torch.sin(alt1_rad) * torch.sin(alt2_rad)
    
    # Clamp to valid range for arccos with small epsilon to avoid edge cases
    cosdelta = torch.clamp(cosdelta, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Calculate angular distance in radians
    ang_dist_rad = torch.acos(cosdelta)
    
    # Handle exact edge cases explicitly
    ang_dist_rad[cosdelta == 1.0] = 0.0  # Perfect alignment
    ang_dist_rad[cosdelta == -1.0] = torch.pi  # Opposite directions
    
    # Convert to degrees
    ang_dist_deg = torch.rad2deg(ang_dist_rad)

    # Apply reduction
    if reduction == "sum":
        return ang_dist_rad.sum(), ang_dist_deg.sum()
    elif reduction == "mean":
        return ang_dist_rad.mean(), ang_dist_deg.mean()
    elif reduction == None or reduction == 'None':
        return ang_dist_rad, ang_dist_deg
    else: 
        raise RuntimeError("Reduction not supported: Use sum, mean or None")
    
def AngularError(vec1, vec2, reduction='mean'):
    """
    Compute angular error between 3D direction vectors.
    
    This function calculates the angle between two vectors using the dot product
    formula. Useful for evaluating 3D direction predictions in Cartesian coordinates.
    
    Mathematical Formula:
        cos(θ) = (v1 · v2) / (|v1| |v2|)
        θ = arccos(cos(θ))
    
    Args:
        vec1 (torch.Tensor): First set of vectors with shape (batch_size, 3)
        vec2 (torch.Tensor): Second set of vectors with shape (batch_size, 3)
        reduction (str, optional): Reduction method. Defaults to 'mean'
            Options: 'sum', 'mean', None
            
    Returns:
        tuple: (angle_rad, angle_deg)
            - angle_rad: Angular errors in radians
            - angle_deg: Angular errors in degrees
            
    Raises:
        RuntimeError: If reduction method is not supported
        
    Note:
        - Handles vectors of any magnitude (not necessarily unit vectors)
        - Clamps cosine values to prevent arccos domain errors
        - Returns both radians and degrees
    """
    # Compute dot product for each pair of vectors
    dot_product = torch.sum(vec1 * vec2, dim=1)

    # Compute vector magnitudes (L2 norms)
    norm_vec1 = torch.norm(vec1, dim=1)
    norm_vec2 = torch.norm(vec2, dim=1)

    # Compute cosine of angle between vectors
    cos_theta = dot_product / (norm_vec1 * norm_vec2)

    # Clamp to valid range for arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle in radians
    angle_rad = torch.acos(cos_theta)

    # Convert to degrees
    angle_deg = torch.rad2deg(angle_rad)

    # Apply reduction
    if reduction == "sum":
        return angle_rad.sum(), angle_deg.sum()
    elif reduction == "mean":
        return angle_rad.mean(), angle_deg.mean()
    elif reduction == None or reduction == 'None':
        return angle_rad, angle_deg
    else: 
        raise RuntimeError("Reduction not supported: Use sum, mean or None")


class VectorLoss(nn.Module):
    """
    Angle-based loss for 2D vector predictions.
    
    This loss function penalizes the angular difference between predicted and
    target 2D vectors, regardless of their magnitudes. Useful when direction
    matters more than magnitude.
    
    Attributes:
        alpha (float): Scaling factor for the loss
        reduction (str): Reduction method ('mean' or 'sum')
    """
    
    def __init__(self, alpha=0.001, reduction='mean'):
        """
        Initialize the VectorLoss.
        
        Args:
            alpha (float, optional): Scaling factor. Defaults to 0.001
            reduction (str, optional): Reduction method. Defaults to 'mean'
        """
        super(VectorLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, output, target):
        """
        Compute the vector angle loss.
        
        This method calculates the absolute angular difference between
        output and target vectors, normalized to [0, π].
        
        Args:
            output (torch.Tensor): Predicted vectors with shape (batch_size, 2)
            target (torch.Tensor): Target vectors with shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Angular difference loss
            
        Raises:
            RuntimeError: If reduction method is not 'sum' or 'mean'
        """
        # Calculate angles using atan2 (returns range [-π, π])
        angles_output = torch.atan2(output[:, 1], output[:, 0])
        angles_target = torch.atan2(target[:, 1], target[:, 0])

        # Compute absolute angle difference
        angle_diff = torch.abs(angles_output - angles_target)

        # Normalize to [0, π] range (shortest angular path)
        angle_diff = torch.remainder(angle_diff + torch.pi, 2 * torch.pi) - torch.pi
        angle_diff = torch.abs(angle_diff)

        # Apply reduction
        if self.reduction == "sum":
            return angle_diff.sum()
        elif self.reduction == "mean":
            return angle_diff.mean()
        else: 
            raise RuntimeError("Reduction not supported: Use sum or mean")

def smooth_BCE(eps=0.1):
    """
    Generate smoothed BCE (Binary Cross-Entropy) targets for label smoothing.
    
    Label smoothing prevents the model from becoming overconfident by using
    soft targets instead of hard 0/1 labels. This can improve generalization.
    
    Args:
        eps (float, optional): Smoothing parameter. Defaults to 0.1
            Determines how much to smooth the labels
            
    Returns:
        tuple: (positive_label, negative_label)
            - positive_label: Smoothed value for positive class (1 - 0.5*eps)
            - negative_label: Smoothed value for negative class (0.5*eps)
            
    Example:
        >>> cp, cn = smooth_BCE(eps=0.1)
        >>> cp  # 0.95 instead of 1.0
        >>> cn  # 0.05 instead of 0.0
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


def generate_hot_ones(device, cn, cp, outputs, targets):
    """
    Create one-hot encoded targets with label smoothing.
    
    This function generates soft one-hot encoded targets for multi-class
    classification with label smoothing applied.
    
    Args:
        device: PyTorch device (CPU or CUDA)
        cn (float): Negative class smoothed value
        cp (float): Positive class smoothed value
        outputs (torch.Tensor): Model outputs (used for shape)
        targets (torch.Tensor): Ground truth class indices
        
    Returns:
        torch.Tensor: Smoothed one-hot encoded targets
            Shape: same as outputs
            
    Example:
        >>> outputs = torch.randn(32, 10)  # batch_size=32, num_classes=10
        >>> targets = torch.tensor([3, 7, 1, ...])  # class indices
        >>> cp, cn = smooth_BCE(eps=0.1)
        >>> t = generate_hot_ones(device, cn, cp, outputs, targets)
        >>> # t[0] = [0.05, 0.05, 0.05, 0.95, 0.05, ...]  # class 3 is positive
    """
    # Initialize all values to negative class value
    t = torch.full_like(outputs, cn, device=device)
    n = outputs.shape[0]
    # Set positive class values
    t[range(n), targets] = cp
    return t

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification.
    
    Focal loss down-weights easy examples and focuses training on hard negatives.
    This is particularly useful for highly imbalanced datasets where the model
    can achieve high accuracy by simply predicting the majority class.
    
    Mathematical Formula:
        FL(p_t) = -α_t (1-p_t)^γ log(p_t)
        
        where:
        - p_t: model's estimated probability for the correct class
        - α_t: weighting factor (alpha parameter)
        - γ: focusing parameter (gamma parameter)
    
    Attributes:
        alpha (torch.Tensor or None): Class weights
        gamma (float): Focusing parameter (higher = more focus on hard examples)
        reduction (str): Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize the Focal Loss.
        
        Args:
            alpha (torch.Tensor or None, optional): Class weights. Defaults to None
                If None, all classes weighted equally
                If Tensor, should have shape (num_classes,)
            gamma (float, optional): Focusing parameter. Defaults to 2.0
                0 = equivalent to cross-entropy
                Higher values = more focus on hard examples
            reduction (str, optional): Reduction method. Defaults to 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction 
        
    def set_alpha(self, alpha):
        """
        Update the class weights.
        
        Args:
            alpha (torch.Tensor): New class weights
        """
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        """
        Compute the focal loss.
        
        Args:
            inputs (torch.Tensor): Raw model outputs (logits)
                Shape: (batch_size, num_classes)
            targets (torch.Tensor): Ground truth class indices
                Shape: (batch_size,)
                
        Returns:
            torch.Tensor: Focal loss value
            
        Process:
            1. Compute standard cross-entropy loss
            2. Compute p_t (probability of correct class)
            3. Apply focal term: (1 - p_t)^gamma
            4. Weight by cross-entropy
            5. Apply class weights (alpha) if provided
        """
        # Compute cross-entropy loss (unreduced)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Get probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Apply focal term and weight by CE loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BCELogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with Logits and Label Smoothing.
    
    This loss combines binary cross-entropy with label smoothing for improved
    generalization. It operates on raw logits (pre-sigmoid outputs).
    
    Attributes:
        device: PyTorch device
        label_smoothing (float): Label smoothing parameter
        BCE: Binary cross-entropy loss function
        cp (float): Smoothed positive label value
        cn (float): Smoothed negative label value
    """
    
    def __init__(self, device, cls_pw=1.0, label_smoothing=0.0):
        """
        Initialize the BCE with logits loss.
        
        Args:
            device: PyTorch device (CPU or CUDA)
            cls_pw (float, optional): Positive class weight. Defaults to 1.0
            label_smoothing (float, optional): Label smoothing factor. Defaults to 0.0
        """
        super().__init__()
        self.device = device
        self.label_smoothing = label_smoothing
        self.BCE = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cls_pw, device=self.device)
        )
        # Generate smoothed label values
        self.cp, self.cn = smooth_BCE(eps=self.label_smoothing)

    def forward(self, outputs, targets):
        """
        Compute BCE loss with label smoothing.
        
        Args:
            outputs (torch.Tensor): Raw model outputs (logits)
            targets (torch.Tensor): Ground truth class indices
            
        Returns:
            torch.Tensor: BCE loss value
        """
        # Generate smoothed one-hot targets
        t = generate_hot_ones(self.device, self.cn, self.cp, outputs, targets)
        
        # Compute BCE loss
        bce_loss = self.BCE(outputs, t)
        
        return bce_loss


class EvidClassification():
    """
    Evidential classification using Dirichlet distribution.
    
    This class implements evidential deep learning for classification,
    where the model outputs a Dirichlet distribution over class probabilities
    rather than point estimates. This provides both predictions and uncertainty.
    
    Attributes:
        class_weights (torch.Tensor or None): Optional class weights
    """
    
    def __init__(self, class_weights=None):
        """
        Initialize evidential classification.
        
        Args:
            class_weights (torch.Tensor or None, optional): Class weights.
                Defaults to None
        """
        self.class_weights = class_weights

    def dirichlet_reg(self, alpha, y):
        """
        Compute Dirichlet distribution regularization.
        
        This regularization term encourages the model to output a uniform
        Dirichlet distribution for incorrect classes, preventing overconfident
        wrong predictions.
        
        Args:
            alpha (torch.Tensor): Dirichlet parameters (concentration)
            y (torch.Tensor): One-hot encoded targets
            
        Returns:
            torch.Tensor: KL divergence from uniform Dirichlet
        """
        # Remove evidence from correct class, keep evidence from wrong classes
        alpha = y + (1 - y) * alpha

        # Uniform Dirichlet distribution (target)
        beta = torch.ones_like(alpha)

        # Compute KL divergence between Dirichlet distributions
        sum_alpha = alpha.sum(-1)
        sum_beta = beta.sum(-1)

        t1 = sum_alpha.lgamma() - sum_beta.lgamma()
        t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
        t3 = alpha - beta
        t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

        kl = t1 - t2 + (t3 * t4).sum(-1)
        return kl.sum()

    def dirichlet_mse(self, alpha, y):
        """
        Compute mean squared error for Dirichlet distribution.
        
        This term measures the accuracy of the mean prediction from the
        Dirichlet distribution, accounting for its uncertainty.
        
        Args:
            alpha (torch.Tensor): Dirichlet parameters
            y (torch.Tensor): One-hot encoded targets
            
        Returns:
            torch.Tensor: MSE loss
        """
        # Sum of Dirichlet parameters (total evidence)
        sum_alpha = alpha.sum(-1, keepdims=True)
        
        # Mean prediction (expected probability)
        p = alpha / sum_alpha
        
        # Prediction error term
        t1 = (y - p).pow(2)
        
        # Uncertainty term (variance of Dirichlet)
        t2 = ((p * (1 - p)) / (sum_alpha + 1))

        # Apply class weights if provided
        if self.class_weights is not None:
            t1 = t1 * self.class_weights.unsqueeze(0)
            t2 = t2 * self.class_weights.unsqueeze(0)

        mse = t1 + t2
        return mse.sum()

    def loss(self, alpha, y, lamb=1.0):
        """
        Compute the evidential classification loss.
        
        Combines Dirichlet MSE with regularization to produce predictions
        with calibrated uncertainty estimates.
        
        Args:
            alpha (torch.Tensor): Dirichlet parameters from model
            y (torch.Tensor): Ground truth class indices
            lamb (float, optional): Regularization weight. Defaults to 1.0
            
        Returns:
            torch.Tensor: Combined loss value
        """
        num_classes = alpha.shape[-1]
        # Convert to one-hot encoding
        y = F.one_hot(y, num_classes)
        # Combine MSE and regularization
        return self.dirichlet_mse(alpha, y) + lamb * self.dirichlet_reg(alpha, y)
