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
        cell_indices: Optional[np.ndarray] = None, # Only get cell indicies from this array
        batch_labels: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.distances = distances
        if self.distances.dtype != np.float32:
            self.distances = self.distances.astype(np.float32)
        self.batch_labels = batch_labels
        self.n_genes = X.shape[1]
        if cell_indices is None:
            self.cell_indices = np.arange(X.shape[0])
        else:
            self.cell_indices = cell_indices
            
        if not isinstance(self.cell_indices, np.ndarray):
            self.cell_indices = np.array(self.cell_indices)

        if batch_labels is None:
            self.batch_labels = np.zeros(X.shape[0], dtype = np.int32)
        if not issparse(X):
            self.log_library_size = np.log(np.maximum(X.sum(axis = 1), 1))
        else:
            self.log_library_size = np.log(np.maximum(X.toarray().sum(axis = 1), 1))
        self.max_neighbors = np.diff(distances.indptr).max()

    @classmethod
    def from_graph(
        self,
        graph: SpatialNeighbors,
        cell_indices: Optional[np.ndarray] = None,
        batch_key: Optional[str] = None,
    ):
        X = graph.adata.X
        return SpatialDataset(
            X = X,
            distances = graph.adata.obsp[graph.distance_key],
            cell_indices = cell_indices,
            batch_labels = graph.adata.obs[batch_key].values if batch_key is not None else None,
        )
    
    def __len__(self):
        return len(self.cell_indices)
    
    def __getitem__(self, idx):
        # Central cell index and gene expression
        cell_idx = self.cell_indices[idx]        
        # Batch labels
        batch_label = self.batch_labels[cell_idx]
        
        # Get all neighbor indices/distances
        start = self.distances.indptr[cell_idx]
        end = self.distances.indptr[cell_idx + 1]
        neighbor_idxs_all = self.distances.indices[start:end]
        neighbor_dists_all = self.distances.data[start:end]
        n_neighbors = len(neighbor_idxs_all)
        neighbor_dists = np.zeros(self.max_neighbors, dtype = np.float32)
        neighbor_mask = torch.zeros(self.max_neighbors, dtype = torch.bool)

        neighbor_dists[:n_neighbors] = neighbor_dists_all
        neighbor_mask[:n_neighbors] = True
        
        log_lib_size = self.log_library_size[cell_idx]
        
        return {
            'cell_idx': cell_idx,
            'neighbor_mask': neighbor_mask,
            'distances': torch.from_numpy(neighbor_dists),
            'log_library_size': log_lib_size,
            'batch_label': batch_label
        }

    def __getitems__(self, idxs):
        
        cell_idx = self.cell_indices[idxs]
        batch_label = self.batch_labels[cell_idx]
        
        starts = self.distances.indptr[cell_idx]
        ends = self.distances.indptr[cell_idx + 1]
        max_neighbors = (ends - starts).max()
        #max_neighbors = np.diff(self.distances[cell_idx].indptr).max()

        neighbor_dists = np.zeros((len(idxs), max_neighbors), dtype = np.float32)
        neighbor_mask = np.zeros((len(idxs), max_neighbors), dtype = bool)
        neighbor_idx = np.full((len(idxs), max_neighbors), -1, dtype = np.int32)
        
        for b, idx in enumerate(cell_idx):
            start = self.distances.indptr[idx]
            end = self.distances.indptr[idx + 1]
            neighbor_idxs_all = self.distances.indices[start:end]
            neighbor_dists_all = self.distances.data[start:end]
            
            n_neighbors = len(neighbor_idxs_all)
            if n_neighbors == 0:
                continue
            
            neighbor_dists[b, :n_neighbors] = neighbor_dists_all
            neighbor_mask[b, :n_neighbors] = True
            neighbor_idx[b, :n_neighbors] = neighbor_idxs_all
        
        log_lib_size = self.log_library_size[cell_idx]
        
        return {
            'cell_idx': torch.from_numpy(cell_idx),
            'neighbor_idx': torch.from_numpy(neighbor_idx),
            'neighbor_mask': torch.from_numpy(neighbor_mask),
            'distances': torch.from_numpy(neighbor_dists),
            'log_library_size': torch.from_numpy(log_lib_size),
            'batch_label': torch.from_numpy(batch_label)
        }
