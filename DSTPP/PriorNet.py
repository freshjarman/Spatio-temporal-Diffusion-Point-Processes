"""
PriorNet.py - History-Adaptive Prior Network (HAP) for DSTPP

This module implements a learnable prior distribution conditioned on history representations.
Instead of using a fixed standard Gaussian N(0, I) as the noise prior in Flow Matching,
PriorNet generates event-specific prior distributions N(μ(H), σ²(H)) based on the
historical context H encoded by the Transformer.

HAP (History Adaptive Prior) Design Philosophy:
    - μ(H) is directly supervised by ground truth via NLL Loss (regression to x_real)
    - σ(H) represents calibrated prediction uncertainty, learned via NLL Loss
    - NO KL regularization to N(0,1) - we WANT an informative prior, not standard Gaussian
    - The learned prior provides a better starting point for Flow Matching

Key Design Principles:
    1. Use Xavier initialization for stable training start
    2. Constrain σ to [min_std, max_std] to prevent variance collapse or explosion
    3. NLL Loss supervises both μ (accuracy) and σ (uncertainty calibration)

Usage:
    prior_net = PriorNet(cond_dim=64, output_dim=3)  # 3 = 1(time) + 2(space)
    mu, std = prior_net(history_encoding)  # Get distribution parameters
    z = prior_net.sample(history_encoding)  # Sample from adaptive prior
    nll = prior_net.nll_loss(x_real, history_encoding)  # NLL for supervision

Reference:
    - Conditional Flow Matching (Lipman et al., ICLR 2023)
    - Flow Matching for Generative Modeling (Lipman et al., 2022)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorNet(nn.Module):
    """
    History-Adaptive Prior Network (HAP).
    
    Maps history representations H to Gaussian distribution parameters:
        p(z|H) = N(μ(H), diag(σ²(H)))
    
    HAP Design (History Adaptive Prior):
        - μ is supervised by ground truth via NLL Loss (regression to x_real)
        - σ represents calibrated prediction uncertainty, learned via NLL Loss
        - No KL regularization (we WANT the prior to be informative, not standard Gaussian)
    
    Args:
        cond_dim: Dimension of the conditioning history representation (e.g., d_model=64)
        output_dim: Dimension of the output noise (1 + loc_dim, e.g., 3 for 2D location)
        hidden_dim: Hidden layer dimension in the MLP
        min_std: Minimum standard deviation to prevent variance collapse
        max_std: Maximum standard deviation to prevent numerical instability
    """

    def __init__(
        self,
        cond_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.min_std = min_std
        self.max_std = max_std

        # MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logstd_head = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training start using Xavier initialization."""
        # Mean head: small random initialization for breaking symmetry
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.1)
        nn.init.zeros_(self.mean_head.bias)

        # Log-std head: initialize to output moderate uncertainty
        nn.init.xavier_uniform_(self.logstd_head.weight, gain=0.1)
        nn.init.zeros_(self.logstd_head.bias)

    def forward(self, cond: torch.Tensor) -> tuple:
        """
        Compute distribution parameters from conditioning.
        
        Args:
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            mean: [batch_size, output_dim] distribution mean (prediction of x_real)
            std: [batch_size, output_dim] distribution std (prediction uncertainty)
        """
        # Handle 3D input: [B, 1, cond_dim] -> [B, cond_dim]
        if cond.dim() == 3:
            cond = cond.squeeze(1)

        h = self.backbone(cond)  # [B, hidden_dim]

        mean = self.mean_head(h)  # [B, output_dim]

        # Constrain std to [min_std, max_std] via sigmoid
        logstd_raw = self.logstd_head(h)  # [B, output_dim]
        std = torch.sigmoid(logstd_raw) * (self.max_std - self.min_std) + self.min_std

        return mean, std

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Sample from the adaptive prior using reparameterization.
        
        Args:
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            z: [batch_size, output_dim] sampled noise
        """
        mean, std = self.forward(cond)
        eps = torch.randn_like(mean)
        return mean + std * eps

    def log_prob(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of z under the adaptive prior.
        
        log p(z|H) = -0.5 * [D*log(2π) + sum(log(σ²)) + sum((z-μ)²/σ²)]
        
        Args:
            z: [batch_size, output_dim] samples to evaluate
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            log_prob: [batch_size] log probability
        """
        # Handle 3D z: [B, 1, output_dim] -> [B, output_dim]
        if z.dim() == 3:
            z = z.squeeze(1)

        mean, std = self.forward(cond)
        var = std**2

        log_prob = -0.5 * (self.output_dim * math.log(2 * math.pi) + torch.sum(torch.log(var), dim=-1) + torch.sum((z - mean)**2 / var, dim=-1))
        return log_prob  # [B]

    def log_prob_decomposed(self, z: torch.Tensor, cond: torch.Tensor) -> tuple:
        """
        Compute log probability decomposed into temporal and spatial components.
        
        Args:
            z: [batch_size, output_dim] samples (first dim is temporal)
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            log_prob: [batch_size] total log probability
            log_prob_t: [batch_size] temporal log probability
            log_prob_s: [batch_size] spatial log probability
        """
        if z.dim() == 3:
            z = z.squeeze(1)

        mean, std = self.forward(cond)
        var = std**2

        # Total
        log_prob = -0.5 * (self.output_dim * math.log(2 * math.pi) + torch.sum(torch.log(var), dim=-1) + torch.sum((z - mean)**2 / var, dim=-1))

        # Temporal (first dimension)
        log_prob_t = -0.5 * (math.log(2 * math.pi) + torch.log(var[:, 0]) + (z[:, 0] - mean[:, 0])**2 / var[:, 0])

        # Spatial (remaining dimensions)
        D_s = self.output_dim - 1
        log_prob_s = -0.5 * (D_s * math.log(2 * math.pi) + torch.sum(torch.log(var[:, 1:]), dim=-1) + torch.sum(
            (z[:, 1:] - mean[:, 1:])**2 / var[:, 1:], dim=-1))

        return log_prob, log_prob_t, log_prob_s

    def nll_loss(self, x_real: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute Negative Log-Likelihood loss for supervising PriorNet (HAP core).
        
        NLL = 0.5 * [D*log(2π) + sum(log(σ²)) + sum((x_real - μ)² / σ²)]
        
        This loss:
            - Forces μ to be close to x_real (via the quadratic term)
            - Calibrates σ as prediction uncertainty (via the log(σ) term)
            - σ² in denominator auto-adjusts: large error → larger σ; small error → smaller σ
        
        Args:
            x_real: [batch_size, output_dim] ground truth values (normalized to [-1, 1])
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            nll: [batch_size] NLL values for each sample
        """
        # Handle 3D input
        if x_real.dim() == 3:
            x_real = x_real.squeeze(1)  # [B, 1, D] -> [B, D]

        mean, std = self.forward(cond)
        var = std**2

        # NLL = 0.5 * [D * log(2π) + sum(log(σ²)) + sum((x - μ)² / σ²)]
        nll = 0.5 * (self.output_dim * math.log(2 * math.pi) + torch.sum(torch.log(var), dim=-1) + torch.sum((x_real - mean)**2 / var, dim=-1))

        return nll  # [B]

    def nll_loss_decomposed(self, x_real: torch.Tensor, cond: torch.Tensor) -> tuple:
        """
        Compute NLL loss decomposed into temporal and spatial components.
        
        Args:
            x_real: [batch_size, output_dim] ground truth (first dim is temporal)
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            nll: [batch_size] total NLL
            nll_t: [batch_size] temporal NLL
            nll_s: [batch_size] spatial NLL
        """
        if x_real.dim() == 3:
            x_real = x_real.squeeze(1)

        mean, std = self.forward(cond)
        var = std**2

        # Total NLL
        nll = 0.5 * (self.output_dim * math.log(2 * math.pi) + torch.sum(torch.log(var), dim=-1) + torch.sum((x_real - mean)**2 / var, dim=-1))

        # Temporal NLL (first dimension)
        nll_t = 0.5 * (math.log(2 * math.pi) + torch.log(var[:, 0]) + (x_real[:, 0] - mean[:, 0])**2 / var[:, 0])

        # Spatial NLL (remaining dimensions)
        D_s = self.output_dim - 1
        nll_s = 0.5 * (D_s * math.log(2 * math.pi) + torch.sum(torch.log(var[:, 1:]), dim=-1) + torch.sum(
            (x_real[:, 1:] - mean[:, 1:])**2 / var[:, 1:], dim=-1))

        return nll, nll_t, nll_s

    def kl_divergence(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from adaptive prior to standard Gaussian.
        
        KL(N(μ, σ²) || N(0, 1)) = 0.5 * sum(σ² + μ² - 1 - log(σ²))
        
        NOTE: This method is DEPRECATED in HAP design. We no longer use KL regularization
        because we WANT an informative prior, not one close to N(0, I).
        Kept for backward compatibility and ablation studies.
        
        Args:
            cond: [batch_size, cond_dim] history representation
        
        Returns:
            kl: [batch_size] KL divergence values
        """
        mean, std = self.forward(cond)
        var = std**2

        kl = 0.5 * torch.sum(var + mean**2 - 1 - torch.log(var), dim=-1)
        return kl  # [B]
