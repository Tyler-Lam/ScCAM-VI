import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List, Optional

class MLPDecoder(nn.Module):
    """
    Decoder class to take intrinsic embedding and spatial context and output parameters of ZINB for each gene
    
    Parameters:
    -----------
    latent_dim: int
        Number of latent dimensions as input
    n_genes: int
        Number of genes for output
    attn_dim: Optional[int] = None
        Number of dimensions for for spatial context. If not provided, use latent_dim
    n_batches: int = 1
        Number of unique batches
    batch_dims: int = 8
        Number of dimensions for batch correction
    dropout: float = 0.1
        Dropout percent for training
    activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu'
        Activation functions
    use_layer_norm: bool = True
        Use layer norm for outputs of MLP layers
    """
    def __init__(
        self,
        latent_dim: int,
        n_genes: int,
        attn_dim: Optional[int] = None,
        hidden_dims: List[int] = [],
        n_batches: int = 1,
        batch_dims: int = 8,
        dropout: float = 0.1,
        activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu',
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.use_batch = batch_dims > 0 and n_batches > 1
        self.batch_embed = None
        self.batch_dims = 0
        if self.use_batch:
            self.batch_dims = batch_dims
            self.batch_embed = nn.Embedding(n_batches, self.batch_dims)
        self.latent_dim = latent_dim + batch_dims
        self.attn_dim = latent_dim + batch_dims if attn_dim is None else attn_dim + batch_dims
        self.n_genes = n_genes
        self.hidden_dims = hidden_dims
        self.use_layer_norm = use_layer_norm
        if isinstance(hidden_dims, int):
            self.hidden_dims = [self.hidden_dims]
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope = 0.01)
        }
        self.activation = activations[activation]
        
        # Make the base embedding decoder
        dims = [self.latent_dim] + self.hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.decoder_base = nn.Sequential(*layers)
        self.rho = nn.Sequential(nn.Linear(dims[-1], n_genes), nn.Softmax(dim = -1))
        
        # Make the spatial context decoder
        dims = [self.attn_dim] + self.hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.decoder_spatial = nn.Sequential(*layers)
        self.lfc = nn.Linear(dims[-1], n_genes)
        
        # Gene specific parameters
        self.pi = nn.Parameter(torch.zeros(n_genes))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        
    def forward(self, z: torch.Tensor, context: torch.Tensor, log_library_size: torch.Tensor, batch_label: torch.Tensor, alpha: float = 1.0):
        if self.batch_embed is not None:
            b = self.batch_embed(batch_label)
            zb = torch.cat([z, b])
            cb = torch.cat([context, b])
        else:
            zb = z
            cb = context
        
        # Intrinsic decoder
        h = self.decoder_base(zb)
        rho = self.rho(h)
        
        # Spatial decoder
        h_spatial = self.decoder_spatial(cb)
        lfc = self.lfc(h_spatial)
        
        # Convert intrinsic decoder to expected baseline gene counts
        lib_size = torch.exp(log_library_size).unsqueeze(-1)
        mu_base = lib_size * rho
        # Calculate spatially aware expected gene counts
        mu = mu_base * torch.exp(alpha * lfc)
        # Get gene specific parameters
        pi = F.sigmoid(self.pi)
        theta = torch.exp(self.log_theta)
        return mu, theta, pi, mu_base, lfc