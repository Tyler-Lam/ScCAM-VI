from dataclasses import dataclass, field, asdict
import json
from typing import Literal, Optional, List

def cfgtojson(cfg, fn):
    if not fn.endswith('.json'):
        fn += '.json'
    with open(fn, 'w') as f:
        json.dump(asdict(cfg), f)
        
@dataclass
class NeighborsConfig:
    unique_core_key: str = 'core_id'
    spatial_key: str = 'spatial'
    neighbor_method: Literal['knn', 'radius', 'precomputed'] = 'knn'
    precomputed_neighbor_key: str = 'spatial_connectivities'
    precomputed_distance_key: str = 'spatial_distances'
    radius: float = 100
    max_neighbors: Optional[int] = None
    
@dataclass
class ModelConfig:
    n_genes: int
    latent_dim: int
    hidden_dims: list[int] = field(default_factory = lambda: [])
    num_heads: int = 1
    dropout: float = 0.1
    n_batches: int = 1
    batch_dim: int = 8
    d_min: float = 0
    d_max: float = 1
    rbf_n_basis: int = 16
    rbf_spacing: Literal['linearl', 'log'] = 'log'
    activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu'
    replacement: Literal['mean', 'zero', 'noise'] = 'mean'
    
@dataclass
class TrainerConfig:
    # Neighbor graph kwargs
    unique_core_key: str = 'core_id'
    spatial_key: str = 'spatial'
    neighbor_method: Literal['knn', 'radius', 'precomputed'] = 'knn'
    precomputed_neighbor_key: str = 'spatial_connectivities'
    precomputed_distance_key: str = 'spatial_distances'
    n_neighbors: int = 15
    radius: float = 100
    knn_max_radius: Optional[float] = None
    max_neighbors: int = 20
    
    # Splitting kwargs
    split_method: Literal['grid', 'core'] = 'grid'
    stratify_by: Optional[str] = None
    bin_length: float = 400
    buffer_cells: bool = True
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    
    # Dataset kwargs
    layer: Optional[str] = None

    # Dataloader kwargs
    batch_size: int = 2**10
    num_workers: int = 0

    # Model kwargs
    latent_dim: int = 16
    hidden_dims: List[int] = field(default_factory = lambda: [])
    num_heads: int = 1
    dropout: float = 0.1
    d_min: float = 0
    d_max: float = 1
    rbf_n_basis: int = 16
    rbf_spacing: Literal['linearl', 'log'] = 'log'
    activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu'
    
    # Optimizer kwargs