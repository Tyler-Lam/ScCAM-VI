import numpy as np
import anndata as ad
from scipy.sparse import issparse, csr_matrix
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.model_selection import train_test_split
from typing import Literal, Optional
import warnings
from tqdm import tqdm

from config import *

def get_hex_bins(
    coords: np.ndarray,
    bin_length: float
):
    """
    Bin coordinates in a hex grid and assign grid idx
    Pure vibe coded function but it works for now?
    """
    x = coords[:,0]
    y = coords[:,1]
    
    q = (np.sqrt(3) / 3 * x - 1/3 * y) / bin_length
    r = (2/3 * y) / bin_length

    x = q
    z = r
    y = -x - z

    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)
    
    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)
    
    cond1 = (dx > dy) * (dx > dz)
    cond2 = (~cond1) & (dy > dz)
    
    rx[cond1] = -ry[cond1] - rz[cond1]
    ry[cond2] = -rx[cond2] - rz[cond2]
    rz[~cond1 & ~cond2] = -rx[~cond1 & ~cond2] - ry[~cond1 & ~cond2]
    
    rx = rx.astype(int)
    rz = rz.astype(int)
    
    bin_coords = np.stack([rx, rz], axis = 1)
    unique_bins, bin_idx = np.unique(bin_coords, axis = 0, return_inverse = True)
    
    return bin_idx

def flag_buffer_cells(
    bin_labels: np.ndarray,
    global_idx: np.ndarray,
    neighbors: csr_matrix,
    bin_categories: np.ndarray,
):
    """
    Flag cells as buffer if they have neighbors in different bins
    
    Parameters:
    ------------
    bin_labels: np.ndarray
        Array of bin labels for a single core
    global_idx: np.ndarray
        Global index for each cell
    neighbors: csr_matrix
        Neighborhood graph in global indices
    """
    
    global_to_local = np.full(global_idx.max() + 1, -1, dtype = np.int32)
    global_to_local[global_idx] = np.arange(len(global_idx))
    is_buffer = np.zeros(len(bin_labels), dtype = bool)
    # for each cell, get bin indices of neighbors
    # If neighbors are in different bins, flag as buffer
    for i, b in enumerate(bin_labels):
        # Get neighbor indices
        neighbor_idxs = neighbors[i].indices
        neighbor_idxs = neighbor_idxs[neighbor_idxs != -1]
        neighbor_bin_labels = bin_labels[global_to_local[neighbor_idxs]]
        if np.any(neighbor_bin_labels != b) & np.any(bin_categories[np.unique(neighbor_bin_labels)] != bin_categories[b]):
            is_buffer[i] = True
    return is_buffer
            

class SpatialNeighbors:
    def __init__(
        self,
        adata: ad.AnnData,
        unique_core_key: str = 'core_id',
        spatial_key: str = 'spatial',
        neighbor_method: Literal['radius', 'precomputed'] = 'radius',
        neighbor_key: str = 'spatial_connectivities',
        distance_key: str = 'spatial_distances',
        radius: float = 100,
        max_neighbors: Optional[int] = None,
    ):
        self.n_cells = len(adata)
        self.adata = adata
        self.unique_core_key = unique_core_key
        self.spatial_key = spatial_key
        self.neighbor_method = neighbor_method
        self.neighbor_key = neighbor_key
        self.distance_key = distance_key
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.bin_labels = None
        self.is_buffer = None
        self.neighbors = [None] * self.n_cells
        self.distances = [None] * self.n_cells
        
    def build_neighbors(self):
        if self.neighbor_method == 'precomputed':
            return
        
        cores = self.adata.obs[self.unique_core_key].values
        coords = self.adata.obsm[self.spatial_key]
        
        for core in np.unique(cores):
            core_mask = (cores == core)
            core_global_idx = np.where(core_mask)[0]
            core_coords = coords[core_mask]
            self._build_neighbors_radius(core_coords, core_global_idx)
            
        dists = np.concatenate(self.distances)
        row_ind = np.repeat(
            np.arange(len(self.neighbors)),
            [len(x) for x in self.neighbors]
        )
        col_ind = np.concatenate(self.neighbors)
        self.adata.obsm[self.distance_key] = csr_matrix((dists, (row_ind, col_ind)), shape = (self.n_cells, self.n_cells))
        conn_matrix = self.adata.obsm[self.distance_key].copy()
        conn_matrix.data[:] = 1
        conn_matrix = conn_matrix.astype(bool)
        self.adata.obsm[self.neighbor_key] = conn_matrix
        
    def _build_neighbors_radius(self, coords, global_idx):
        nn = NearestNeighbors(radius = self.radius, algorithm = 'ball_tree')
        nn.fit(coords)
        dists, indices = nn.radius_neighbors(coords, return_distance = True, sort_results = True)
        
        for i, (row_dist, row_idx) in enumerate(zip(dists, indices)):
            mask = (row_idx != i)
            #n = min(len(row_idx) - 1, self.max_neighbors)
            self.neighbors[global_idx[i]] = global_idx[row_idx[mask]][:self.max_neighbors]
            self.distances[global_idx[i]] = row_dist[mask][:self.max_neighbors]
            
    def get_neighbors(self, cell_idx: int):
        return self.adata.obsm[self.distance_key]
    
    def _bin_coords(
        self,
        bin_length: float,
        verbose: bool = False,
    ):
        
        bin_labels = np.full(len(self.adata), -1, dtype = np.int32)
        
        cores = self.adata.obs[self.unique_core_key].values
        coords = self.adata.obsm[self.spatial_key]
        
        for core in tqdm(np.unique(cores), desc = "Binning data for splitting", disable = not verbose):
            core_mask = (cores == core)
            core_global_idx = np.where(core_mask)[0]
            core_coords = coords[core_mask]
            
            bin_labels_core = get_hex_bins(core_coords, bin_length)
            bin_labels[core_mask] = bin_labels_core

        self.bin_labels = bin_labels

    def split_data(
        self,
        method: Literal['grid', 'core'] = 'grid',
        bin_length: float = 200,
        buffer_cells: bool = True,
        test_size: float = 0.15,
        val_size: float = 0.15,
        stratify_by: Optional[str] = None,
        random_state: float = 42,
        verbose: bool = False
    ):
        if method == 'grid':
            split_idx = self.split_by_grid(
                bin_length = bin_length,
                buffer_cells = buffer_cells,
                test_size = test_size,
                val_size = val_size,
                random_state = random_state,
                verbose = verbose
            )
            self.adata.obs['bin_labels'] = self.bin_labels
            self.adata.obs['split_category'] = None
            self.adata.obs.iloc[self.train_idx, self.adata.obs.columns.get_loc('split_category')] = 'train'
            self.adata.obs.iloc[self.test_idx, self.adata.obs.columns.get_loc('split_category')] = 'test'
            self.adata.obs.iloc[self.val_idx, self.adata.obs.columns.get_loc('split_category')] = 'val'
            if self.is_buffer is not None:
                self.adata.obs['is_buffer'] = self.is_buffer
                self.adata.obs.iloc[self.is_buffer, self.adata.obs.columns.get_loc('split_category')] = 'buffer'

        elif method == 'core':
            split_idx = self.split_by_core(
                val_size = val_size,
                test_size = test_size,
                stratify_by = stratify_by,
                random_state = random_state
            )
            self.adata.obs['split_category'] = None
            self.adata.obs.iloc[self.train_idx, self.adata.obs.columns.get_loc('split_category')] = 'train'
            self.adata.obs.iloc[self.test_idx, self.adata.obs.columns.get_loc('split_category')] = 'test'
            self.adata.obs.iloc[self.val_idx, self.adata.obs.columns.get_loc('split_category')] = 'val'
        else:
            raise ValueError(f"Split method must be one of ['bin', 'core']. Received {method}")
        self.split_idx = split_idx
        return split_idx

    def split_by_grid(
        self,
        bin_length: float,
        buffer_cells: bool = True,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: float = 42,
        verbose: bool = False
    ):
        
        if self.bin_labels is None:
            self._bin_coords(bin_length, verbose)
            
        train_idx, val_idx, test_idx = [], [], []
        cores = self.adata.obs[self.unique_core_key].values
        is_buffer = np.zeros(len(self.adata), dtype = bool)
        for core in tqdm(np.unique(cores), desc = "Splitting cores", disable = not verbose):
            core_mask = (cores == core)
            core_global_idx = np.where(core_mask)[0]
            
            bin_labels = self.bin_labels[core_mask]
            bin_labels_unique = np.unique(bin_labels)
            train_val_bins, test_bins = train_test_split(bin_labels_unique, test_size = test_size, random_state = random_state)
            train_bins, val_bins = train_test_split(train_val_bins, test_size = val_size / (1 - test_size), random_state = random_state)
            
            bin_categories = np.zeros(bin_labels_unique.max() + 1, dtype = np.int8)
            bin_categories[train_bins] = 1
            bin_categories[val_bins] = 2
            bin_categories[test_bins] = 3
            
            if buffer_cells:
                
                is_buffer_core = flag_buffer_cells(
                    bin_labels,
                    core_global_idx,
                    self.adata.obsm[self.neighbor_key][core_mask],
                    bin_categories
                )
                
                is_buffer[core_mask] = is_buffer_core
            else:
                is_buffer_core = np.zeros(len(bin_labels), dtype = bool)
            
            train_mask = np.isin(bin_labels, train_bins) & ~is_buffer_core
            train_idx.extend(core_global_idx[train_mask])
            
            val_mask = np.isin(bin_labels, val_bins) & ~is_buffer_core
            val_idx.extend(core_global_idx[val_mask])
            
            test_mask = np.isin(bin_labels, test_bins) & ~is_buffer_core
            test_idx.extend(core_global_idx[test_mask])
            
        self.is_buffer = is_buffer
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.val_idx = val_idx
        return {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'val_idx': val_idx
        }
        
    def split_by_core(
        self,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_by: Optional[str] = None,
        random_state: int = 42
    ):
        
        cores = self.adata.obs[self.unique_core_key].unique()
        
        if stratify_by is not None:
            core_strata = adata.obs.groupby(self.unique_core_key)[stratify_by].agg(lambda x: x.mode()[0]).to_dict()
            
            train_val_cores, test_cores = train_test_split(core_strata.index.values, test_size = test_size, stratify = core_strata, random_state = random_state)
            train_cores, val_cores = train_test_split(train_val_cores, test_size = val_size / (1 - test_size), stratify = core_strata.loc[train_val_cores].values, random_state = random_state)
            
        else:
            train_val_cores, test_cores = train_test_split(cores, test_size = test_size, random_state = random_state)
            train_cores, val_cores = train_test_split(train_val_cores, test_size = val_size / (1 - test_size), random_state = random_state)
            
        train_idx = np.where(self.adata.obs[self.unique_core_key].isin(train_cores))[0]
        test_idx = np.where(self.adata.obs[self.unique_core_key].isin(test_cores))[0]
        val_idx = np.where(self.adata.obs[self.unique_core_key].isin(val_cores))[0]
        
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.val_idx = val_idx
        return {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'val_idx': val_idx
        }
        
    def slice_by_index(
        self,
        cell_indices: np.ndarray
    ):
        
        all_indices = set(self.adata.obsm[self.neighbor_key][cell_indices].indices)
        all_indices = sorted(all_indices)
        global_to_local = np.full(self.n_cells, -1, dtype = np.int64)
        global_to_local[all_indices] = np.arange(len(all_indices))
        global_to_local = np.concatenate([global_to_local, [-1]])
        
        sliced = SpatialNeighbors.__new__(SpatialNeighbors)
        sliced.n_cells = len(all_indices)
        sliced.max_neighbors = self.max_neighbors
        sliced.unique_core_key = self.unique_core_key
        sliced.spatial_key = self.spatial_key
        sliced.neighbor_method = self.neighbor_method
        sliced.neighbor_key = self.neighbor_key
        sliced.distance_key = self.distance_key
        sliced.radius = self.radius
        
        sliced.adata = self.adata[all_indices].copy()
        sliced.adata.obsm[self.distance_key].indices = global_to_local[sliced.adata.obsm[self.distance_key].indices]
        sliced.adata.obsm[self.neighbor_key].indices = global_to_local[sliced.adata.obsm[self.neighbor_key].indices]
        return sliced, global_to_local[cell_indices]
