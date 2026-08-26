import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
                
    def kl_divergence(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        mu_prior: torch.Tensor,
    ):
        kl = 0.5 * (
            log_var.exp() + (mu - mu_prior).pow(2) - 1 - log_var
        ).sum(dim = -1)
        return kl.mean()
    
    def zinb_nll(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
        eps: float = 1e-8,
    ):
        theta = theta.unsqueeze(0)
        pi = pi.unsqueeze(0)
        mu = mu.clamp_min(eps)
        log_nb = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
            + theta * torch.log(theta / (theta + mu))
            + x * torch.log(mu / (theta + mu ))
        )
        log_nb_zero = theta * torch.log( theta / (theta + mu) )
        
        log_zero = torch.log(pi + eps) + log_nb_zero
        log_nonzero = torch.log(1 - pi + eps) + log_nb
        
        is_zero = (x == 0).float()
        
        ll = (is_zero * log_zero) + (1 - is_zero) * log_nonzero

        return -ll
        
        
    def forward(
        self,
        mu_z_prior: torch.Tensor,
        z: torch.Tensor,
        log_var: torch.Tensor,
        mu: torch.Tensor,
        mu_intrinsic: torch.Tensor,
        delta: torch.Tensor,
        pi: torch.Tensor,
        theta: torch.Tensor,
        x: torch.Tensor,
        beta_kl: float = 1.0,
        gamma: float = 1.0,
        lambda_mu_prior: float = 1e-3,
        lambda_delta: float = 1e-3,
    ):
        # KL Divergence on latent space
        loss_kl = self.kl_divergence(z, log_var, mu_z_prior)
        
        # L1 regularization on latent means priors
        loss_prior_reg = mu_z_prior.abs().mean()
        
        # zinb nll loss
        loss_recon = self.zinb_nll(x, mu, theta, pi)
        
        # zinb nll loss on intrinsic predictions
        loss_recon_intrinsic = self.zinb_nll(x, mu_intrinsic, theta, pi)
        
        # L1 regularization on log fold changes
        loss_delta_reg = delta.abs().sum(dim = -1).mean()
        
        # Total loss
        loss = loss_recon.sum(dim = -1).mean() + gamma * loss_recon_intrinsic.sum(dim = -1).mean() + beta_kl * loss_kl + loss_kl.mean() + lambda_mu_prior * loss_prior_reg + lambda_delta * loss_delta_reg
        
        return {
            'loss': loss,
            'loss_recon': loss_recon.sum(dim = -1).mean().item(),
            'loss_recon_intrinsic': loss_recon_intrinsic.sum(dim = -1).mean().item(),
            'loss_kl': loss_kl.mean().item(),
            'loss_prior_reg': loss_prior_reg.item(),
            'loss_delta_reg': loss_delta_reg.item()
        }
