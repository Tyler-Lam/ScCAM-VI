import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal, Tuple
import torch
import torch.nn as nn
import numpy as np
from typing import Literal

class RBFDistanceEncoder(nn.Module):
    def __init__(
        self,
        n_basis: int = 16,
        d_min: float = 0,
        d_max: float = 1,
        spacing: Literal['log', 'linear'] = 'linear',
    ):
        super().__init__()
        
        self.n_basis = n_basis
        self.d_min = d_min
        self.d_max = d_max
        self.spacing = spacing

        if spacing == 'log':
            centers = torch.logspace(
                np.log10(self.d_min + 1e-5),
                np.log10(self.d_max + 1e-5),
                self.n_basis
            )
        elif spacing == 'linear':
            centers = torch.linspace(
                self.d_min,
                self.d_max,
                self.n_basis
            )
        else:
            raise ValueError(f"Invalid spacing option given: {self.spacing}. Must be 'linear' or 'log'")
        
        widths = torch.zeros(n_basis)
        widths[0] = max(centers[1] - centers[0], centers[0])
        widths[1:-1] = centers[2:] - centers[1:-1]
        widths[-1] = widths[-2]
        
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths)
        
        self.weights = nn.Parameter(torch.ones(n_basis) / n_basis)
            
    def forward(self, distances: torch.Tensor):
        d = distances.unsqueeze(-1)
        activations = torch.exp(
            -((d - self.centers) ** 2) / (2 * self.widths ** 2 + 1e-8)
        )
        bias = activations @ self.weights
        return bias
    
    def get_distance_curve(self):
        centers = self.centers.detach().cpu().numpy()
        widths = self.widths.detach().cpu().numpy()
        weights = self.weights.detach().cpu().numpy()

        return centers, widths, weights
    
    
class SpatialCrossAttention(nn.Module):
    
    def __init__(
        self,
        latent_dim: int,
        attn_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.1,
        d_min: float = 0,
        d_max: float = 1,
        rbf_n_basis: int = 16,
        rbf_spacing: Literal['log', 'linear'] = 'log',
        need_weights: bool = True,
        project_inputs: bool = True,
        topk: int = -1,
    ):
        super().__init__()
        

        self.latent_dim = latent_dim
        self.attn_dim = latent_dim if attn_dim is None else attn_dim
        if self.attn_dim % num_heads != 0:
            raise ValueError(f'Attention dimension ({self.attn_dim}) must be divisible by num_heads ({num_heads})')
        if self.attn_dim != self.latent_dim and not project_inputs:
            raise ValueError(f'project_inputs must be True if latent_dim ({self.latent_dim}) != attn_dim ({self.attn_dim})')
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_min = d_min
        self.d_max = d_max
        self.rbf_n_basis = rbf_n_basis
        self.rbf_spacing = rbf_spacing
        self.need_weights = need_weights
        self.project_inputs = project_inputs
        self.topk = topk
        
        self.query_proj = None
        self.key_proj = None
        self.value_proj = None
        if self.project_inputs:
            self.query_proj = nn.Linear(self.latent_dim, self.attn_dim)
            self.key_proj = nn.Linear(self.latent_dim, self.attn_dim)
            self.value_proj = nn.Linear(self.latent_dim, self.attn_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim = self.attn_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )
        
        self.distance_encoding = RBFDistanceEncoder(
            n_basis = self.rbf_n_basis,
            d_min = self.d_min,
            d_max = self.d_max,
            spacing = self.rbf_spacing,
        )
        
        self.layer_norm = nn.LayerNorm(self.attn_dim)
        
    def forward(
        self,
        central_z: torch.Tensor,
        neighbor_z: torch.Tensor,
        neighbor_mask: torch.Tensor,
        distances: torch.Tensor
    ):
        if self.project_inputs:
            query = self.query_proj(central_z).unsqueeze(1)
            key = self.key_proj(neighbor_z)
            value = self.value_proj(neighbor_z)
        else:
            query = central_z.unsqueeze(1)
            key = neighbor_z
            value = neighbor_z

        padding_mask = torch.zeros(
            neighbor_mask.shape[0], 1, neighbor_mask.shape[1], dtype = torch.float32, device = central_z.device
        )
        padding_mask = padding_mask.masked_fill(
            ~neighbor_mask.unsqueeze(1), float('-inf')
        )
        attn_bias = padding_mask + self.distance_encoding(distances).unsqueeze(1)
        attn_bias = attn_bias.expand(-1, self.num_heads, -1)
        
        attn_bias_flat = attn_bias.reshape(
            attn_bias.shape[0] * self.num_heads, 1, attn_bias.shape[-1]
        )
        
        if self.topk != -1:
            B, L, D = query.shape
            S = key.shape[1]
            with torch.no_grad():
                _, attn_weights = self.attention(
                    query = query,
                    key = key,
                    value = value,
                    attn_mask = attn_bias_flat,
                    need_weights = True,
                    average_attn_weights = False
                )

                topk_idx = attn_weights.topk(self.topk, dim = -1).indices
                keep = torch.zeros_like(attn_weights, dtype = torch.bool)
                keep.scatter_(-1, topk_idx, True)
                block_mask = (~keep).squeeze(2)
                
            attn_bias = attn_bias.masked_fill(block_mask, float('-inf'))
            
        attn_bias = attn_bias.reshape(
            attn_bias.shape[0] * self.num_heads, 1, attn_bias.shape[-1]
        )
        
        context, attn_weights = self.attention(
            query = query,
            key = key,
            value = value,
            attn_mask = attn_bias,
            need_weights = False,
            average_attn_weights = False
        )
        
        context = context.squeeze(1)
        return self.layer_norm(context), attn_weights