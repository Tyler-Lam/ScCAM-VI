from neighbors import *
from autoencoder import *
from training import *
import warnings
from itertools import combinations
from scipy.special import factorial
from scipy.stats import spearmanr
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.colors as colors

class SpatialAblation:
    def __init__(
        self,
        trainer: SpatialAutoencoderTrainer,
        batch_size: Optional[int] = None
    ):
        
        self.model = trainer.model
        self.adata = trainer.adata
        self.layer = trainer.layer
        self.device = trainer.device
        self.batch_size = batch_size if batch_size is not None else len(self.adata)
        self.model.eval()
        self.embedding = trainer.get_embedding()['z_intrinsic'] if trainer.embedding is None else trainer.embedding['z_intrinsic']
        self.prediction = trainer.predict()
        self.prediction['mu'] = np.log(1 + self.prediction['mu'] / self.prediction['mu'].sum(axis = 1)[:,np.newaxis] * 1e4)
        self.prediction['mu_intrinsic'] = np.log(1 + self.prediction['mu_intrinsic'] / self.prediction['mu_intrinsic'].sum(axis = 1)[:,np.newaxis] * 1e4)
        self.mean_z = torch.from_numpy(self.embedding.mean(axis = 0))
        self.null_z = torch.zeros(self.mean_z.shape, dtype = self.mean_z.dtype)
        # Make dataset object starting from latent embedding
        self.dataset = SpatialDataset(
            X = self.embedding,
            distances = trainer.adata.obsp[trainer.graph.distance_key],
            celltype_labels = None if trainer.celltype_key is None else trainer.adata.obs[f'{trainer.celltype_key}_int'].values,
            batch_labels = None if trainer.batch_key is None else trainer.adata.obs[f'{trainer.batch_key}_int'].values,
        )
        self.dataset.log_library_size = trainer.dataset.log_library_size
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            num_workers = 0,
            shuffle = False,
            collate_fn = lambda x: x,
        )
        
        self.embedding_tensor = torch.from_numpy(self.embedding)

    @torch.no_grad()
    def _run_with_mask(
        self,
        ablation_idxs: np.ndarray, # mask of cells to ablate
        method: Literal['mask', 'mean', 'prior'] = 'mask',
        show_progress: bool = False
    ):
        self.model.eval()
        out = []
        
        for batch in self.dataloader:
            cell_idx = batch['cell_idx']
            neighbor_idx = batch['neighbor_idx']
            neighbor_mask = batch['neighbor_mask']
            distances = batch['distances'].to(self.device)
            log_library_size = batch['log_library_size'].to(self.device)
            batch_label = batch['batch_label'].to(self.device)
            
            z_intrinsic = self.embedding_tensor[cell_idx].to(self.device)
            
            # Do neighbor and ablation mask calculations on cpu, then move to cpu
            neighbor_z = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1], z_intrinsic.shape[1]),
                dtype = z_intrinsic.dtype
            )
            neighbor_z[neighbor_mask] = self.embedding_tensor[neighbor_idx[neighbor_mask]]
            
            ablation_mask = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1]),
                dtype = torch.bool
            )
            
            ablation_mask[neighbor_mask] = torch.isin(neighbor_idx[neighbor_mask], torch.from_numpy(ablation_idxs))

            if method == 'mask':
                neighbor_mask[ablation_mask] = False
            elif method == 'mean':
                neighbor_z[ablation_mask] = self.mean_z
            else:
                raise ValueError(f'Invalid ablation method given: {method}')
            
            # Move to gpu
            neighbor_mask = neighbor_mask.to(self.device)
            z_intrinsic = z_intrinsic.to(self.device)
            neighbor_z = neighbor_z.to(self.device)
    
            z_spatial = torch.zeros((z_intrinsic.shape[0], self.model.attn_dim), dtype = torch.float32, device = z_intrinsic.device)
            
            has_neighbors = neighbor_mask.any(dim = -1)
            if has_neighbors.any():
                context, weights = self.model.attention(
                    central_z = z_intrinsic[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                z_spatial[has_neighbors] = context
    
            mu, theta, pi, mu_intrinsic, delta = self.model.decode(z_intrinsic, z_spatial, log_library_size, batch_label)
            out.append(delta.cpu().numpy())
        
        out = np.concatenate(out)
        return out

    @torch.no_grad()
    def _run_ablation_permutation(
        self,
        ablation_idxs: np.ndarray,
        permutation: np.ndarray,
    ):
        self.model.eval()
        out = []
        
        permutation = torch.from_numpy(permutation)
        ablation_idxs = torch.from_numpy(ablation_idxs)
        
        for batch in self.dataloader:
            cell_idx = batch['cell_idx']
            neighbor_idx = batch['neighbor_idx']
            neighbor_mask = batch['neighbor_mask']
            distances = batch['distances'].to(self.device)
            log_library_size = batch['log_library_size'].to(self.device)
            batch_label = batch['batch_label'].to(self.device)
            
            z_intrinsic = self.embedding_tensor[cell_idx].to(self.device)
            
            ablation_mask = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1]),
                dtype = torch.bool
            )
            
            ablation_mask[neighbor_mask] = torch.isin(neighbor_idx[neighbor_mask], ablation_idxs)
            neighbor_idx[ablation_mask] = permutation[neighbor_idx[ablation_mask]]
            
            # Do neighbor and ablation mask calculations on cpu, then move to cpu
            neighbor_z = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1], z_intrinsic.shape[1]),
                dtype = z_intrinsic.dtype
            )
            neighbor_z[neighbor_mask] = self.embedding_tensor[neighbor_idx[neighbor_mask]]
            
            # Move to gpu
            neighbor_mask = neighbor_mask.to(self.device)
            z_intrinsic = z_intrinsic.to(self.device)
            neighbor_z = neighbor_z.to(self.device)
            z_spatial = torch.zeros((z_intrinsic.shape[0], self.model.attn_dim), dtype = torch.float32, device = z_intrinsic.device)

            has_neighbors = neighbor_mask.any(dim = -1)
            if has_neighbors.any():
                context, weights = self.model.attention(
                    central_z = z_intrinsic[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                z_spatial[has_neighbors] = context
            
            mu, theta, pi, mu_intrinsic, delta = self.model.decode(z_intrinsic, z_spatial, log_library_size, batch_label)
            out.append(delta.cpu().numpy())
            
        out = np.concatenate(out)
        return out
    
    @torch.no_grad()
    def _run_shap_coalition(
        self,
        coalition: np.ndarray,
        category: str,
        method: Literal['mask', 'mean', 'zero'] = 'mean',
    ):
        self.model.eval()
        scores = {c: [] for c in coalition}
        
        for batch in self.dataloader:
            cell_idx = batch['cell_idx']
            neighbor_idx = batch['neighbor_idx']
            neighbor_mask = batch['neighbor_mask']
            distances = batch['distances'].to(self.device)
            log_library_size = batch['log_library_size'].to(self.device)
            batch_label = batch['batch_label'].to(self.device)
            
            z_intrinsic = self.embedding_tensor[cell_idx].to(self.device)
            
            # Do neighbor and ablation mask calculations on cpu, then move to cpu
            neighbor_z = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1], z_intrinsic.shape[1]),
                dtype = z_intrinsic.dtype
            )
            neighbor_z[neighbor_mask] = self.embedding_tensor[neighbor_idx[neighbor_mask]]
            
            z_intrinsic = z_intrinsic.to(self.device)
            neighbor_z = neighbor_z.to(self.device)
            neighbor_idx = neighbor_idx.to(self.device)
            neighbor_mask = neighbor_mask.to(self.device)
    
            z_spatial = torch.zeros((z_intrinsic.shape[0], self.model.attn_dim), dtype = torch.float32, device = z_intrinsic.device)
            
            has_neighbors = neighbor_mask.any(dim = -1)
            if has_neighbors.any():
                context, weights = self.model.attention(
                    central_z = z_intrinsic[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                z_spatial[has_neighbors] = context
            
            mu, theta, pi, mu_intrinsic, prev_delta = self.model.decode(z_intrinsic, z_spatial, log_library_size, batch_label)
            
            for c in coalition:
                # Get the ablation mask for the coalition
                ablation_idxs = torch.from_numpy(np.where(self.adata.obs[category] == c)[0]).to(self.device)
                ablation_mask = torch.zeros(
                    (cell_idx.shape[0], neighbor_idx.shape[1]),
                    dtype = torch.bool,
                    device = self.device
                )
                ablation_mask[neighbor_mask] = torch.isin(neighbor_idx[neighbor_mask], ablation_idxs)
                # Do the ablation
                if method == 'mask':
                    neighbor_mask[ablation_mask] = False
                elif method == 'mean':
                    neighbor_z[ablation_mask] = self.mean_z.to(self.device)
                else:
                    raise ValueError(f'Invalid ablation method given: {method}')
                has_neighbors = neighbor_mask.any(dim = -1)
                z_spatial = torch.zeros((z_intrinsic.shape[0], self.model.attn_dim), dtype = torch.float32, device = z_intrinsic.device)
                if has_neighbors.any():
                    context, weights = self.model.attention(
                        central_z = z_intrinsic[has_neighbors],
                        neighbor_z = neighbor_z[has_neighbors],
                        neighbor_mask = neighbor_mask[has_neighbors],
                        distances = distances[has_neighbors],
                    )
                    z_spatial[has_neighbors] = context
                
                mu, theta, pi, mu_intrinsic, delta = self.model.decode(z_intrinsic, z_spatial, log_library_size, batch_label)
                scores[c].append((prev_delta - delta).cpu().numpy())
                prev_delta = delta
        scores = {c: np.concatenate(scores[c]) for c in scores}
        return scores

    def run_ablation_mc(
        self,
        category: str,
        n_iter: int = 1000,
        method: Literal['mask', 'mean', 'prior'] = 'mask',
        random_state: int = 42
    ):
        rng = np.random.default_rng(random_state)
        categories_unique = self.adata.obs[category].unique()
        
        scores = {cat: np.zeros(self.adata.shape) for cat in categories_unique}
        
        for i in tqdm(range(n_iter), desc = "Calculating MC SHAP scores"):
            coalition = rng.permutation(categories_unique)
            scores_this_iter = self._run_shap_coalition(coalition = coalition, category = category, method = method)
            for c in scores_this_iter:
                scores[c] += scores_this_iter[c]
        
        scores = {c: scores[c] / n_iter for c in scores}
        return scores
    
    
    def get_scores_df(
        self,
        scores: dict[np.ndarray],
        category: str,
        central_category: str,
        neighbor_category: str,
        connectivities_key: str = 'spatial_connectivities',
        min_neighbors: int = 1,
    ):
        central_mask = (self.adata.obs[category] == central_category).values
        central_idx = np.where(central_mask)[0]
        neighbor_mask = (self.adata.obs[category] == neighbor_category).values
        neighbor_idx = np.where(neighbor_mask)[0]

        n_neighbors = np.zeros(self.adata.shape[0], dtype = np.int32)
        for i in central_idx:
            start = self.adata.obsp[connectivities_key].indptr[i]
            end = self.adata.obsp[connectivities_key].indptr[i + 1]
            n_neighbors[i] = neighbor_mask[self.adata.obsp[connectivities_key].indices[start:end]].sum()

        n_neighbor_mask = n_neighbors >= min_neighbors
        
        means = scores[neighbor_category][central_mask & n_neighbor_mask].mean(axis = 0)
        stds = scores[neighbor_category][central_mask & n_neighbor_mask].std(axis = 0)
        medians = np.median(scores[neighbor_category][central_mask & n_neighbor_mask], axis = 0)
        out_df = pd.DataFrame({'mean': means, 'std': stds, 'median': medians}, index = self.adata.var_names)
        
        return out_df
    
    
    def plot_scores(
        self,
        scores: dict[np.ndarray],
        gene: str,
        category: str,
        central_category: str,
        connectivities_key: str = 'spatial_connectivities',
        min_neighbors: int = 1,
        x_label: str = 'x_centroid',
        y_label: str = 'y_centroid',
        method: Literal['mask', 'mean', 'zero'] = 'mean',
        log: bool = False
    ):
        gene_idx = np.where(self.adata.var_names == gene)[0][0]
        central_mask = (self.adata.obs[category] == central_category).values
        central_idx = np.where(central_mask)[0]
        bdata = self.adata[central_mask].copy()
        s = 120000 / bdata.shape[0]
        n_celltypes = len(self.adata.obs[category].unique())
        fig, ax = plt.subplots(n_celltypes+1, 3, figsize = (24, 4 * n_celltypes), width_ratios = [1, 1, 1], constrained_layout = True)
        fig.suptitle(f"{central_category}")
        #mappable = ax[0,0].scatter(
        #    x = bdata.obs[x_label],
        #    y = bdata.obs[y_label],
        #    c = bdata[:,gene_idx].X.toarray(),
        #    s = s,
        #    linewidths = 0,
        #    cmap = 'viridis',
        #    edgecolors = 'none',
        #)
        #ax[0,0].set_aspect("equal")
        #ax[0,0].set_title(f"Observed {gene} count")
        #fig.colorbar(mappable, ax = ax[0,0], label = 'Observed Counts', pad = 0.001)
        
        mu = self.prediction['mu']
        mappable = ax[0,0].scatter(
            x = bdata.obs[x_label],
            y = bdata.obs[y_label],
            c = mu[central_mask, gene_idx],
            vmin = np.percentile(mu[central_mask, gene_idx], 1),
            vmax = np.percentile(mu[central_mask, gene_idx], 95),
            s = s,
            linewidths = 0,
            cmap = 'viridis',
            edgecolors = 'none',
        )
        ax[0,0].set_aspect("equal")
        ax[0,0].set_title(f"{gene} prediction")
        fig.colorbar(mappable, ax = ax[0,0], label = 'Predicted Counts', pad = 0.001)
        
        mu_intrinsic = self.prediction['mu_intrinsic']
        mappable = ax[0,1].scatter(
            x = bdata.obs[x_label],
            y = bdata.obs[y_label],
            c = mu_intrinsic[central_mask, gene_idx],
            vmin = np.percentile(mu_intrinsic[central_mask, gene_idx], 1),
            vmax = np.percentile(mu_intrinsic[central_mask, gene_idx], 95),
            s = s,
            linewidths = 0,
            cmap = 'viridis',
            edgecolors = 'none',
        )
        ax[0,1].set_aspect("equal")
        ax[0,1].set_title(f"Basline {gene} prediction")
        fig.colorbar(mappable, ax = ax[0,1], label = 'Predicted Counts', pad = 0.001)
        
        lfcs = self.prediction['delta']
        mappable = ax[0,2].scatter(
            x = bdata.obs[x_label],
            y = bdata.obs[y_label],
            c = lfcs[central_mask, gene_idx],
            vmin = np.percentile(lfcs[central_mask, gene_idx], 1),
            vmax = np.percentile(lfcs[central_mask, gene_idx], 99),
            s = s,
            linewidths = 0,
            cmap = 'coolwarm',
            edgecolors = 'none',
        )
        ax[0,2].set_aspect("equal")
        ax[0,2].set_title(f"LFC from baseline")
        fig.colorbar(mappable, ax = ax[0,2], label = 'LFC from baseline', pad = 0.001)
        
        scores_all = np.concatenate([scores[c][:,gene_idx] for c in scores])
        vmin = np.percentile(scores_all, 5)
        vmax = np.percentile(scores_all, 95)
        vmax_sym = max(abs(vmin), abs(vmax))
        symnorm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-vmax_sym, vmax=vmax_sym, base=10)
        counts = bdata[:,gene_idx].X.toarray()

        for n, neighbor_category in enumerate(self.adata.obs[category].unique()):
        
            neighbor_mask = (self.adata.obs[category] == neighbor_category).values
            neighbor_idx = np.where(neighbor_mask)[0]
            
            n_neighbors = np.zeros(self.adata.shape[0], dtype = np.int32)
            for i in central_idx:
                start = self.adata.obsp[connectivities_key].indptr[i]
                end = self.adata.obsp[connectivities_key].indptr[i + 1]
                n_neighbors[i] = neighbor_mask[self.adata.obsp[connectivities_key].indices[start:end]].sum()

            bdata.obs['n_neighbors'] = n_neighbors[central_mask]
            s = 120000 / bdata.shape[0]
            
            mappable = ax[n+1,0].scatter(
                x = bdata.obs[x_label],
                y = bdata.obs[y_label],
                c = counts,
                #vmax = np.percentile(counts, 95),
                s = s,
                linewidths = 0,
                cmap = 'viridis',
                edgecolors = 'none',
            )
            ax[n+1,0].set_aspect("equal")
            ax[n+1,0].set_title(f"Observed {gene} count")
            fig.colorbar(mappable, ax = ax[n+1,0], label = 'Observed Counts', pad = 0.001)
            
            # Plotting number of neighbors
            mappable = ax[n+1,1].scatter(
                x = bdata.obs[x_label],
                y = bdata.obs[y_label],
                c = bdata.obs['n_neighbors'],
                norm = colors.SymLogNorm(linthresh = 1),
                s = s,
                linewidths = 0,
                cmap = 'viridis',
                edgecolors = 'none',
            )
            ax[n+1,1].set_aspect("equal")
            ax[n+1,1].set_title(f"Neighboring {neighbor_category}")
            fig.colorbar(mappable, ax = ax[n+1,1], label = 'n_neighbors', pad = 0.001)

            # Potting scatterplot by shap score
            neighbor_scores = scores[neighbor_category][central_mask][bdata.obs['n_neighbors'] >= min_neighbors, gene_idx]
            mappable = ax[n+1,2].scatter(
                x = bdata.obs[x_label],
                y = bdata.obs[y_label],
                c = scores[neighbor_category][central_mask][:,gene_idx],
                norm = symnorm,
                s = s,
                cmap = 'coolwarm',
                edgecolors = 'none'
            )
            fig.colorbar(mappable, ax = ax[n+1,2], label = "SHAP score", pad = 0.001)
            ax[n+1,2].set_aspect("equal")
            ax[n+1,2].set_title(f'{gene} SHAP scores for {neighbor_category}')
            
            #r = spearmanr(scores[neighbor_category][central_mask][:,gene_idx], bdata.obs['n_neighbors'].values).statistic
            #ax[n+1,2].scatter(
            #    x = scores[neighbor_category][central_mask][:,gene_idx],
            #    y = bdata.obs['n_neighbors'],
            #    s = s / 4,
            #)
            #ax[n+1,2].set_xlim(-vmax_sym, vmax_sym)
            #ax[n+1,2].set_xlabel("SHAP score")
            #ax[n+1,2].set_ylabel("N_neighbors")
            #ax[n+1,2].set_title(f'N_neighbors vs {neighbor_category} SHAP score (spearman r = {r:.3f})')
            #ax[n+1,2].grid()
            
        return fig