import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal
import pandas as pd

from config import *
from encoder import *
from attention import *
from decoder import *
from loss import *

class SpatialAutoEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: List[int] = [],
        num_heads: int = 1,
        dropout: float = 0.1,
        n_batches: int = 1,
        batch_dim: int = 8,
        d_min: float = 0,
        d_max: float = 1,
        rbf_n_basis: int = 16,
        rbf_spacing: Literal['linear', 'log'] = 'log',
        activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu',
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_batches = n_batches
        self.batch_dim = batch_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.d_min = d_min
        self.d_max = d_max
        self.rbf_n_basis = rbf_n_basis
        self.rbf_spacing = rbf_spacing
        self.activation = activation
        
        if isinstance(hidden_dims, int):
            self.hidden_dims = [self.hidden_dims]

        self.encoder = MLPEncoder(
            n_genes = self.n_genes,
            latent_dim = self.latent_dim,
            hidden_dims = self.hidden_dims,
            dropout = self.dropout,
            activation = activation
        )
        
        self.attention = SpatialCrossAttention(
            latent_dim = self.latent_dim,
            num_heads = self.num_heads,
            dropout = self.dropout,
            d_min = self.d_min,
            d_max = self.d_max,
            rbf_n_basis = self.rbf_n_basis,
            rbf_spacing = self.rbf_spacing
        )
        
        self.decoder = MLPDecoder(
            latent_dim = self.latent_dim,
            n_genes = self.n_genes,
            hidden_dims = self.hidden_dims[::-1],
            n_batches = self.n_batches,
            batch_dims = self.batch_dim,
            dropout = self.dropout,
            activation = self.activation,
        )
        
        self.delta = nn.Parameter(torch.zeros(self.latent_dim))
        
    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(
        self,
        central_X: torch.Tensor,
        neighbor_X: torch.Tensor,
        neighbor_mask: torch.Tensor,
        distances: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 1.0,
        mask_pct: float = 0.0,
    ):

        mu, log_var = self.encoder(central_X)
        pre_attn_z = self.reparameterize(mu, log_var)
        batch_size, max_neighbors, n_genes = neighbor_X.shape
        
        neighbor_X_flat = neighbor_X.view(-1, n_genes)
        neighbor_mu_flat, neighbor_log_var_flat = self.encoder(neighbor_X_flat)
        neighbor_z_flat = self.reparameterize(neighbor_mu_flat, neighbor_log_var_flat)
        neighbor_z = neighbor_z_flat.view(batch_size, max_neighbors, self.latent_dim)
        
        # Mask cells with no neighbors from attention
        has_neighbors = neighbor_mask.any(dim = -1)
        post_attn_z = pre_attn_z.clone()
        weights = None
        if has_neighbors.any():
            context, weights = self.attention(
                central_z = pre_attn_z[has_neighbors],
                neighbor_z = neighbor_z[has_neighbors],
                neighbor_mask = neighbor_mask[has_neighbors],
                distances = distances[has_neighbors]
            )
            
            mask = torch.bernoulli(
                torch.full((pre_attn_z.shape[0], 1), 1 - mask_pct, device=pre_attn_z.device)
            )
            
            post_attn_z[has_neighbors] *= mask[has_neighbors]
            post_attn_z[has_neighbors] *= alpha
            post_attn_z[has_neighbors] += gamma * torch.sigmoid(self.delta) * context
        return mu, log_var, pre_attn_z, post_attn_z, weights
    
    def decode(self,post_attn_z: torch.Tensor, log_library_size: torch.Tensor, batch_labels: Optional[torch.Tensor] = None):
        return self.decoder(post_attn_z, log_library_size, batch_labels)
    
    def forward(
        self,
        central_X: torch.Tensor,
        neighbor_X: torch.Tensor,
        neighbor_mask: torch.Tensor,
        distances: torch.Tensor,
        log_library_size: torch.Tensor,
        batch_label: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 1.0,
        mask_pct: float = 0.0
    ):
        mu_z, log_var, pre_attn_z, post_attn_z, attn_weights = self.encode(
            central_X = central_X,
            neighbor_X = neighbor_X,
            neighbor_mask = neighbor_mask,
            distances = distances,
            alpha = alpha,
            gamma = gamma,
            mask_pct = mask_pct
        )
        
        mu_x, theta, pi = self.decode(post_attn_z, log_library_size, batch_label)
        
        return {
            'mu_z': mu_z,
            'log_var': log_var,
            'mu_x' : mu_x,
            'theta': theta,
            'pi': pi,
            'pre_attn_z': pre_attn_z,
            'post_attn_z': post_attn_z,
            'attn_weights': attn_weights
        }
        
    def save(
        self,
        dst_dir: str = '',
        savenm: str = ''
    ):
        cfg = ModelConfig(
            n_genes = self.n_genes,
            latent_dim = self.latent_dim,
            hidden_dims = self.hidden_dims,
            num_heads = self.num_heads,
            dropout = self.dropout,
            d_min = float(self.d_min),
            d_max = float(self.d_max),
            rbf_n_basis = self.rbf_n_basis,
            rbf_spacing = self.rbf_spacing,
            activation = self.activation
        )
        
        sfx = f'_{savenm}' if savenm else ''
        cfgtojson(cfg, f'{dst_dir}/cfg{sfx}')
        torch.save(self.state_dict(), f'{dst_dir}/model{sfx}.pt')