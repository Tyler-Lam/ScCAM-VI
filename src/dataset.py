import numpy as np
import anndata as ad
import scanpy as sc
import torch
from neighbors import *
from torch.utils.data import Dataset
from scipy.sparse import issparse, csr_matrix
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from typing import Optional, Literal, Sequence
import warnings

class SpatialDataset(Dataset):
    def __init__(
        self,
        X: csr_matrix | np.ndarray,
        distances: csr_matrix,
        max_neighbors: int = 50,
        cell_indices: Optional[np.ndarray] = None, # Only get cell indicies from this array
        batch_labels: Optional[np.ndarray] = None,
        random_state: int = 42,
        train: bool = False
    ):
        super().__init__()
        self.distances = distances
        if self.distances.dtype != np.float32:
            self.distances = self.distances.astype(np.float32)
        self.batch_labels = batch_labels
        self.X = X
        if issparse(self.X):
            self.X = self.X.toarray()
        self.n_genes = self.X.shape[1]
        if cell_indices is None:
            self.cell_indices = np.arange(self.X.shape[0])
        else:
            self.cell_indices = cell_indices
        if batch_labels is None:
            self.batch_labels = np.zeros(self.X.shape[0], dtype = np.int32)
        self.log_library_size = np.log(np.maximum(self.X.sum(axis = 1), 1))
        self.train = train
        self.max_neighbors = max_neighbors if self.train else np.diff(distances.indptr).max()
        self.rng = np.random.default_rng(seed = random_state)

    @classmethod
    def from_graph(
        self,
        graph: SpatialNeighbors,
        layer: Optional[str] = None,
        cell_indices: Optional[np.ndarray] = None,
        batch_key: Optional[str] = None,
        max_neighbors: int = 50,
        random_state: int = 42,
        train: bool = False
    ):
        X = graph.adata.X if layer is None else graph.adata.layers[layer]
        return SpatialDataset(
            X = X,
            distances = graph.adata.obsp[graph.distance_key],
            max_neighbors = max_neighbors,
            cell_indices = cell_indices,
            batch_labels = graph.adata.obs[batch_key].values if batch_key is not None else None,
            train = train,
            random_state = random_state
        )
    
    def __len__(self):
        return len(self.cell_indices)
    
    def __getitem__(self, idx):
        # Central cell index and gene expression
        cell_idx = self.cell_indices[idx]
        cell_X = torch.from_numpy(self.X[cell_idx])
        
        # Batch labels
        batch_label = self.batch_labels[cell_idx]
        
        # Get all neighbor indices/distances
        start = self.distances.indptr[idx]
        end = self.distances.indptr[idx + 1]
        neighbor_idxs_all = self.distances.indices[start:end]
        neighbor_dists_all = self.distances.data[start:end]
        n_neighbors = len(neighbor_idxs_all)
        neighbor_dists = np.zeros(self.max_neighbors, dtype = np.float32)
        neighbor_mask = torch.zeros(self.max_neighbors, dtype = torch.bool)
        neighbor_X = np.zeros((self.max_neighbors, self.n_genes), dtype = np.float32)
        if n_neighbors > self.max_neighbors:
            weights = 1 / (neighbor_dists_all + 1)
            weights /= weights.sum()
            chosen = self.rng.choice(np.arange(n_neighbors), size = self.max_neighbors, p = weights)
            
            neighbor_dists = neighbor_dists_all[chosen]
            neighbor_mask[:] = True
            neighbor_X = self.X[neighbor_idxs_all[chosen]]
        else:
            neighbor_dists[:n_neighbors] = neighbor_dists_all
            neighbor_mask[:n_neighbors] = True
            neighbor_X[:n_neighbors] = self.X[neighbor_idxs_all]
            
        log_lib_size = torch.tensor(self.log_library_size[cell_idx], dtype = torch.float32)
        
        return {
            'cell_idx': cell_idx,
            'cell_X': cell_X,
            'neighbor_X': torch.from_numpy(neighbor_X),
            'neighbor_mask': neighbor_mask,
            'distances': torch.from_numpy(neighbor_dists).to(torch.float32),
            'log_library_size': log_lib_size,
            'batch_label': batch_label
        }