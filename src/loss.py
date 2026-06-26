import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        loss_fn: Literal['mse', 'huber', 'zinb'] = 'zinb',
        delta: float = 1.0,
        mask_weight: float = 0.5,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.delta = delta
        self.mask_weight = mask_weight
                
    def kl_divergence(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ):
        kl = 0.5 * (
            log_var.exp() + mu.pow(2) - 1 - log_var
        ).sum(dim = -1)
        return kl.mean()
            
    def mse_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor
    ):
        sq_err = (x_hat - x) ** 2
        if self.gene_weights is not None:
            sq_error = sq_error * self.gene_weights.unsqueeze(0)
        
        return sq_error.mean()
    
    def huber_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor
    ):
        loss = F.huber_loss(x_hat, x, reduction = 'none', delta = self.delta)
        if self.gene_weights is not None:
            loss *= self.gene_weights.unsqueeze(0)
        return loss
    
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
        mu_z: torch.Tensor,
        log_var: torch.Tensor,
        mu_x: torch.Tensor,
        pi: torch.Tensor,
        theta: torch.Tensor,
        x: torch.Tensor,
        beta_kl: float = 1.0,
        gene_mask: Optional[torch.Tensor] = None,
    ):
        loss_kl = self.kl_divergence(mu_z, log_var)
        if self.loss_fn == 'mse':
            loss_recon = self.mse_loss(mu_x, x)
        elif self.loss_fn == 'huber':
            loss_recon = self.huber_loss(mu_x, x)
        elif self.loss_fn == 'zinb':
            loss_recon = self.zinb_nll(x, mu_x, theta, pi)
        else:
            raise ValueError(f'Loss function must be one of ["mse", "huber", "zinb"]. Given {self.loss_fn}')
        
        if gene_mask is not None:
            loss_recon[gene_mask] *= self.mask_weight
            #loss_recon = torch.where(gene_mask, loss_recon * self.mask_weight, loss_recon)
        return loss_recon.sum(dim = -1).mean() + beta_kl * loss_kl.mean(), loss_recon.sum(dim = -1).mean().item(), loss_kl.mean().item()
