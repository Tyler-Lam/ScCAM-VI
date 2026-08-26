import torch
import torch.nn as nn
from typing import List, Optional, Literal

class MLPEncoder(nn.Module):
    """
    Module that passes raw count data through an MLP encoder to produce the mean and log_var of each latent dimension
    
    Parameters:
    -----------
    n_genes: int
        Input number of genes
    latent_dim: int
        Number of latent dimensions
    hidden_dims: List[int] = []
        Dimensions for each hidden layer
    dropout: float = 0.1
        Percent dropout for training
    activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu'
        Activation function
    use_layer_norm: bool = True
        Use layer norm after each layer
    """
    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: List[int] = [],
        dropout: float = 0.1,
        activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu',
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.dropout = dropout
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope = 0.01)
        }
        self.activation = activations[activation]
        self.hidden_dims = hidden_dims
        if isinstance(hidden_dims, int):
            self.hidden_dims = [self.hidden_dims]
        dims = [n_genes] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(self.activation)
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(dims[-1], self.latent_dim)
        self.log_var = nn.Linear(dims[-1], self.latent_dim)
        
    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var