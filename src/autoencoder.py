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

class CellTypePrior(nn.Module):
    def __init__(self, n_celltypes: int, latent_dim: int):
        super().__init__()
        self.prior_mu = nn.Embedding(n_celltypes, latent_dim)
        nn.init.zeros_(self.prior_mu.weight)

    def forward(self, celltype_idx):
        return self.prior_mu(celltype_idx)
    
class SpatialAutoEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        n_celltypes: int = 1,
        hidden_dims: List[int] = [],
        attn_dim: Optional[int] = None,
        project_inputs: bool = True,
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
        self.attn_dim = attn_dim
        self.project_inputs = project_inputs
        self.n_celltypes = n_celltypes
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
            
        self.celltype_prior = CellTypePrior(
            n_celltypes = n_celltypes,
            latent_dim = latent_dim
        )

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
            attn_dim = self.attn_dim,
            project_inputs = self.project_inputs,
            dropout = self.dropout,
            d_min = self.d_min,
            d_max = self.d_max,
            rbf_n_basis = self.rbf_n_basis,
            rbf_spacing = self.rbf_spacing
        )
        
        self.decoder = MLPDecoder(
            latent_dim = self.latent_dim,
            n_genes = self.n_genes,
            attn_dim = self.attn_dim,
            hidden_dims = self.hidden_dims[::-1],
            n_batches = self.n_batches,
            batch_dims = self.batch_dim,
            dropout = self.dropout,
            activation = self.activation,
        )
        
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(self.latent_dim, self.latent_dim)
        
        # Vector gate for attention model
        self.gamma = nn.Parameter(torch.zeros(self.latent_dim))
        
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
    ):

        mu_z, log_var = self.encoder(central_X)
        z_intrinsic = self.reparameterize(mu_z, log_var)

        batch_size, max_neighbors, n_genes = neighbor_X.shape
        
        neighbor_X_flat = neighbor_X.view(-1, n_genes)
        neighbor_z_flat, _ = self.encoder(neighbor_X_flat)
        neighbor_z = neighbor_z_flat.view(batch_size, max_neighbors, self.latent_dim).detach()
        
        # Mask cells with no neighbors from attention
        has_neighbors = neighbor_mask.any(dim = -1)
        z_spatial = torch.zeros((z_intrinsic.shape[0], self.attn_dim), dtype = torch.float32, device = z_intrinsic.device)
        weights = None
        if has_neighbors.any():
            query = z_intrinsic[has_neighbors].detach()
            context, weights = self.attention(
                central_z = query,
                neighbor_z = neighbor_z,
                neighbor_mask = neighbor_mask[has_neighbors],
                distances = distances[has_neighbors]
            )

            z_spatial[has_neighbors] = context
        
        return mu_z, log_var, z_intrinsic, z_spatial, weights
    
    def decode(self, z_intrinsic: torch.Tensor, z_spatial: torch.Tensor, log_library_size: torch.Tensor, batch_labels: Optional[torch.Tensor] = None, alpha = 1.0):
        return self.decoder(z_intrinsic, z_spatial, log_library_size, batch_labels, alpha = alpha)
    
    def forward(
        self,
        central_X: torch.Tensor,
        neighbor_X: torch.Tensor,
        neighbor_mask: torch.Tensor,
        distances: torch.Tensor,
        log_library_size: torch.Tensor,
        batch_label: torch.Tensor,
        celltype_label: torch.Tensor,
        alpha: float = 1.0,
    ):

        mu_z_prior = self.celltype_prior(celltype_label)
        
        mu_z, log_var, z_intrinsic, z_spatial, attn_weights = self.encode(
            central_X = central_X,
            neighbor_X = neighbor_X,
            neighbor_mask = neighbor_mask,
            distances = distances,
        )

        mu, theta, pi, mu_intrinsic, lfc = self.decode(z_intrinsic, z_spatial, log_library_size, batch_label, alpha = alpha)
        
        return {
            'mu_z_prior': mu_z_prior,
            'mu_z': mu_z,
            'log_var': log_var,
            'mu' : mu,
            'theta': theta,
            'pi': pi,
            'mu_intrinsic': mu_intrinsic,
            'lfc': lfc,
            'z_spatial': z_spatial,
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