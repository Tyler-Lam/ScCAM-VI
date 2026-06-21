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
        neighbors: np.ndarray,
        distances: np.ndarray,
        max_neighbors: int = 20,
        cell_indices: Optional[np.ndarray] = None, # Only get cell indicies from this array
        batch_labels: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.neighbors = neighbors
        self.distances = distances
        self.max_neighbors = max_neighbors
        self.batch_labels = batch_labels
        if batch_labels is None:
            self.batch_labels = np.zeros(len(self.neighbors), dtype = np.int32)
        self.X = X
        if issparse(self.X):
            self.X = self.X.toarray()
        self.n_genes = self.X.shape[1]
        if cell_indices is None:
            self.cell_indices = np.arange(self.X.shape[0])
        else:
            self.cell_indices = cell_indices
        self.log_library_size = np.log(np.maximum(self.X.sum(axis = 1), 1))
        self.neighbor_buffer = np.zeros((self.max_neighbors, self.n_genes), dtype = np.float32)
    
    @classmethod
    def from_graph(
        self,
        graph: SpatialNeighbors,
        layer: Optional[str] = None,
        cell_indices: Optional[np.ndarray] = None,
        batch_key: Optional[str] = None,
    ):
        X = graph.adata.X if layer is None else graph.adata.layers[layer]
        return SpatialDataset(
            X = X,
            neighbors = graph.neighbors,
            distances = graph.distances,
            max_neighbors=graph.max_neighbors,
            cell_indices=cell_indices,
            batch_labels = graph.adata.obs[batch_key].values if batch_key is not None else None
        )
    
    def __len__(self):
        return len(self.cell_indices)
    
    def __getitem__(self, idx):
        
        cell_idx = self.cell_indices[idx]
        cell_X = torch.from_numpy(self.X[cell_idx])
        batch_label = self.batch_labels[cell_idx]
        neighbor_idxs = self.neighbors[cell_idx]
        neighbor_mask = neighbor_idxs > -1
        self.neighbor_buffer[:] = 0.0
        self.neighbor_buffer[neighbor_mask] = self.X[neighbor_idxs[neighbor_mask]]
        neighbor_X = torch.from_numpy(self.neighbor_buffer.copy())
        neighbor_dists = torch.from_numpy(self.distances[cell_idx])
        log_lib_size = torch.tensor(self.log_library_size[cell_idx], dtype = torch.float32)
        neighbor_mask_tensor = torch.from_numpy(neighbor_mask)
        return {
            'cell_idx': cell_idx,
            'cell_X': cell_X,
            'neighbor_X': neighbor_X,
            'neighbor_mask': neighbor_mask_tensor,
            'distances': neighbor_dists,
            'log_library_size': log_lib_size,
            'batch_label': batch_label
        }