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
    neighbors: np.ndarray,
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
        neighbor_idxs = neighbors[i]
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
        neighbor_method: Literal['knn', 'radius', 'precomputed'] = 'knn',
        precomputed_neighbor_key: str = 'spatial_connectivities',
        precomputed_distance_key: str = 'spatial_distances',
        n_neighbors: int = 15,
        radius: float = 100,
        knn_max_radius: Optional[float] = None,
        max_neighbors: int = 20,
    ):
        self.n_cells = len(adata)
        self.adata = adata
        self.unique_core_key = unique_core_key
        self.spatial_key = spatial_key
        self.neighbor_method = neighbor_method
        self.precomputed_neighbor_key = precomputed_neighbor_key
        self.precomputed_distance_key = precomputed_distance_key
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.knn_max_radius = knn_max_radius
        self.max_neighbors = max_neighbors
        self.bin_labels = None
        self.is_buffer = None
        if self.max_neighbors < self.n_neighbors:
            warnings.warn("Max neighbors must be >= n_neighbors. Setting max_neighbors = n_neighbors", UserWarning)
            self.max_neighbors = self.n_neighbors
        
        self.neighbors = np.full((self.n_cells, self.max_neighbors), fill_value = -1, dtype = np.int64)
        self.distances = np.zeros((self.n_cells, self.max_neighbors), dtype = np.float32)
        
    def build_neighbors(self):
        if self.neighbor_method == 'precomputed':
            self._build_neighbors_precomputed()
            return
        
        cores = self.adata.obs[self.unique_core_key].values
        coords = self.adata.obsm[self.spatial_key]
        
        for core in np.unique(cores):
            core_mask = (cores == core)
            core_global_idx = np.where(core_mask)[0]
            core_coords = coords[core_mask]
            
            if self.neighbor_method == 'knn':
                self._build_neighbors_knn(core_coords, core_global_idx)
            elif self.neighbor_method == 'radius':
                self._build_neighbors_radius(core_coords, core_global_idx)
        
    def _build_neighbors_precomputed(self):
        if self.precomputed_neighbor_key not in adata.obsp:
            raise ValueError(f"Key {self.precomputed_neighbor_key} is not in adata.obsp")
        if self.precomputed_distance_key not in adata.obsp:
            raise ValueError(f"Key {self.precomputed_distance_key} is not in adata.obsp")
        
        conn = self.adata.obsp[self.precomputed_neighbors_key]
        if isinstance(conn, np.ndarray):
            conn = csr_matrix(conn)
        elif issparse(conn) and not isinstance(conn, csr_matrix):
            conn = conn.tocsr()
            
        dists = self.adata.obsp[self.precomputed_distance_key]
        if isinstance(dists, np.ndarray):
            dists = csr_matrix(dists)
        elif issparse(dists) and not isinstance(dists, csr_matrix):
            dists = dists.tocsr()
            
        for i in range(self.n_cells):
            row = conn.getrow(i)
            idxs = row.indices
            n = min(len(idxs), self.max_neighbors)
            self.neighbors[i, :n] = idxs[:n]
            
            dist_row = dists.getrow(i)
            dist_idxs = row.indices
            n = min(len(dist_idxs), self.max_neighbors)
            self.distances[i, :n] = dist_idxs[:n]
        
    def _build_neighbors_knn(self, coords, global_idx):
        k = min(self.n_neighbors + 1, len(coords))
        nn = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree')
        nn.fit(coords)
        dists, indices = nn.kneighbors(coords)

        for i, (row_dist, row_idx) in enumerate(zip(dists, indices)):
            mask = (row_idx != i)
            if self.knn_max_radius is not None:
                mask = mask & (row_dist <= self.knn_max_radius) 
            n = sum(mask)
            self.neighbors[global_idx[i], :n] = global_idx[row_idx[mask]]
            self.distances[global_idx[i], :n] = row_dist[mask]
            
    def _build_neighbors_radius(self, coords, global_idx):
        nn = NearestNeighbors(radius = self.radius, algorithm = 'ball_tree')
        nn.fit(coords)
        dists, indices = nn.radius_neighbors(coords, return_distance = True, sort_results = True)
        
        for i, (row_dist, row_idx) in enumerate(zip(dists, indices)):
            mask = (row_idx != i)
            n = min(len(row_idx) - 1, self.max_neighbors)
            self.neighbors[global_idx[i], :n] = global_idx[row_idx[mask]][:n]
            self.distances[global_idx[i], :n] = row_dist[mask][:n]
            
    def get_neighbors(self, cell_idx: int):
        mask = (self.neighbors[cell_idx] != -1)
        return (self.neighbors[cell_idx][mask], self.distances[cell_idx][mask])
    
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

    def split_by_bin(
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
                    self.neighbors[core_mask],
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
        
        all_indices = set(cell_indices)
        for i in cell_indices:
            nn = self.get_neighbors(i)[0]
            all_indices.update(nn)
        
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
        sliced.precomputed_neighbor_key = self.precomputed_neighbor_key
        sliced.precomputed_distance_key = self.precomputed_distance_key
        sliced.n_neighbors = self.n_neighbors
        sliced.radius = self.radius
        sliced.knn_max_radius = self.knn_max_radius
        
        sliced.adata = self.adata[all_indices].copy()
        sliced.distances = self.distances[all_indices].copy()
        sliced.neighbors = global_to_local[self.neighbors[all_indices]]
        return sliced, global_to_local[cell_indices]
    
    def get_adata(self):
        cfg = NeighborsConfig(
            unique_core_key = self.unique_core_key,
            spatial_key = self.spatial_key,
            neighbor_method = self.neighbor_method,
            precomputed_distance_key = self.precomputed_distance_key,
            precomputed_neighbor_key = self.precomputed_neighbor_key,
            n_neighbors = self.n_neighbors,
            radius = self.radius,
            knn_max_radius = self.knn_max_radius,
            max_neighbors = self.max_neighbors
        )
        
        self.adata.uns['spatial_neighbors'] = asdict(cfg)
        #cfgtojson(cfg, f'{dst_dir}/cfg_{savenm}.json')
        
        rows, cols, dists, conn = [], [], [], []
        
        for cell_idx in range(self.n_cells):
            neighbors = self.neighbors[cell_idx]
            distances = self.distances[cell_idx]
            
            mask = (neighbors != -1)
            
            for neighbor_idx, dist in zip(neighbors[mask], distances[mask]):
                rows.append(cell_idx)
                cols.append(neighbor_idx)
                dists.append(dist)
                conn.append(1.0)
        
        dist_matrix = csr_matrix(
            (dists, (rows, cols)), shape = (self.n_cells, self.n_cells)
        )
        
        conn_matrix = csr_matrix(
            (conn, (rows, cols)), shape = (self.n_cells, self.n_cells)
        )
        
        self.adata.obsp['distances'] = dist_matrix
        self.adata.obsp['connectivities'] = conn_matrix
        self.adata.uns['neighbors'] = asdict(cfg)
        if self.bin_labels is not None:
            self.adata.obs['bin_labels'] = self.bin_labels
        if self.is_buffer is not None:
            self.adata.obs['is_buffer'] = self.is_buffer
            
        self.adata.obs['split_category'] = None
        self.adata.obs.iloc[self.train_idx, self.adata.obs.columns.get_loc('split_category')] = 'train'
        self.adata.obs.iloc[self.test_idx, self.adata.obs.columns.get_loc('split_category')] = 'test'
        self.adata.obs.iloc[self.val_idx, self.adata.obs.columns.get_loc('split_category')] = 'val'
        if self.is_buffer is not None:
            self.adata.obs.iloc[self.is_buffer, self.adata.obs.columns.get_loc('split_category')] = 'buffer'

        return self.adata