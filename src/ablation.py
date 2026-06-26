from neighbors import *
from autoencoder import *
from training import *
import warnings
from itertools import combinations
from scipy.special import factorial
from torch.cuda.amp import autocast
from tqdm import tqdm

class SpatialAblation:
    def __init__(
        self,
        trainer: SpatialAutoencoderTrainer,
        batch_size: Optional[int] = None
    ):
        
        self.model = trainer.model
        self.adata = trainer.adata
        self.device = trainer.device
        self.batch_size = batch_size if batch_size is not None else len(self.adata)
        self.model.eval()
        self.embedding = trainer.get_embedding()['pre_attention'] if trainer.embedding is None else trainer.embedding['pre_attention']
        self.mean_z = torch.from_numpy(self.embedding.mean(axis = 0))
        self.null_z = torch.zeros(self.mean_z.shape, dtype = self.mean_z.dtype)
        # Make dataset object starting from latent embedding
        self.dataset = SpatialDataset(
            X = self.embedding,
            distances = trainer.adata.obsp[trainer.graph.distance_key],
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
        method: Literal['mask', 'mean', 'zero'] = 'mean',
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
            
            cell_z = self.embedding_tensor[cell_idx].to(self.device)
            
            # Do neighbor and ablation mask calculations on cpu, then move to cpu
            neighbor_z = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1], cell_z.shape[1]),
                dtype = cell_z.dtype
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
            elif method == 'zero':
                neighbor_z[ablation_mask] = self.null_z
            else:
                raise ValueError(f'Invalid ablation method given: {method}')
            
            # Move to gpu
            neighbor_mask = neighbor_mask.to(self.device)
            cell_z = cell_z.to(self.device)
            neighbor_z = neighbor_z.to(self.device)
            post_attn_z = cell_z.clone()

            has_neighbors = neighbor_mask.any(dim = -1)
            if has_neighbors.any():
                context, weights = self.model.attention(
                    central_z = cell_z[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                post_attn_z[has_neighbors] += torch.sigmoid(self.model.gamma) * context
            
            mu_x, theta, pi = self.model.decode(post_attn_z, log_library_size, batch_label)
            out.append(mu_x.cpu().numpy())
        
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
            
            cell_z = self.embedding_tensor[cell_idx].to(self.device)
            
            ablation_mask = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1]),
                dtype = torch.bool
            )
            
            ablation_mask[neighbor_mask] = torch.isin(neighbor_idx[neighbor_mask], ablation_idxs)
            neighbor_idx[ablation_mask] = permutation[neighbor_idx[ablation_mask]]
            
            # Do neighbor and ablation mask calculations on cpu, then move to cpu
            neighbor_z = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1], cell_z.shape[1]),
                dtype = cell_z.dtype
            )
            neighbor_z[neighbor_mask] = self.embedding_tensor[neighbor_idx[neighbor_mask]]
            
            # Move to gpu
            neighbor_mask = neighbor_mask.to(self.device)
            cell_z = cell_z.to(self.device)
            neighbor_z = neighbor_z.to(self.device)
            post_attn_z = cell_z.clone()

            has_neighbors = neighbor_mask.any(dim = -1)
            if has_neighbors.any():
                context, weights = self.model.attention(
                    central_z = cell_z[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                post_attn_z[has_neighbors] += torch.sigmoid(self.model.gamma) * context
            
            mu_x, theta, pi = self.model.decode(post_attn_z, log_library_size, batch_label)
            out.append(mu_x.cpu().numpy())
        
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
            
            cell_z = self.embedding_tensor[cell_idx].to(self.device)
            
            # Do neighbor and ablation mask calculations on cpu, then move to cpu
            neighbor_z = torch.zeros(
                (cell_idx.shape[0], neighbor_idx.shape[1], cell_z.shape[1]),
                dtype = cell_z.dtype
            )
            neighbor_z[neighbor_mask] = self.embedding_tensor[neighbor_idx[neighbor_mask]]
            
            cell_z = cell_z.to(self.device)
            neighbor_z = neighbor_z.to(self.device)
            neighbor_idx = neighbor_idx.to(self.device)
            neighbor_mask = neighbor_mask.to(self.device)
            post_attn_z = cell_z.clone()
            
            has_neighbors = neighbor_mask.any(dim = -1)
            if has_neighbors.any():
                context, weights = self.model.attention(
                    central_z = cell_z[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                post_attn_z[has_neighbors] += torch.sigmoid(self.model.gamma) * context
            
            prev_x, *_ = self.model.decode(post_attn_z, log_library_size, batch_label)
            
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
                elif method == 'zero':
                    neighbor_z[ablation_mask] = self.null_z.to(self.device)
                else:
                    raise ValueError(f'Invalid ablation method given: {method}')
                has_neighbors = neighbor_mask.any(dim = -1)
                post_attn_z = cell_z.clone()
                if has_neighbors.any():
                    context, weights = self.model.attention(
                        central_z = cell_z[has_neighbors],
                        neighbor_z = neighbor_z[has_neighbors],
                        neighbor_mask = neighbor_mask[has_neighbors],
                        distances = distances[has_neighbors],
                    )
                    post_attn_z[has_neighbors] += torch.sigmoid(self.model.gamma) * context
                
                current_x, *_ = self.model.decode(post_attn_z, log_library_size, batch_label)
                scores[c].append((prev_x - current_x).cpu().numpy())
                prev_x = current_x
        scores = {c: np.concatenate(scores[c]) for c in scores}
        return scores

    def run_ablation_mc(
        self,
        category: str,
        n_iter: int = 1000,
        method: Literal['mask', 'mean', 'zero'] = 'mean',
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