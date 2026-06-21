from neighbors import *
from autoencoder import *
from training import *
import warnings
from itertools import combinations
from scipy.special import factorial
from torch.cuda.amp import autocast

class SpatialAblation:
    def __init__(
        self,
        trainer: SpatialAutoencoderTrainer,
        batch_size: Optional[int] = None
    ):
        self.model = trainer.model
        self.device = trainer.device
        self.max_neighbors = trainer.max_neighbors
        self.adata = trainer.adata
        self.batch_size = batch_size if batch_size is not None else len(trainer.adata)
        self.model.eval()
        self.embedding = trainer.get_embedding() if trainer.embedding is None else trainer.embedding
        self.mean_z = torch.from_numpy(self.embedding['pre_attention'].mean(axis = 0))
        
        self.X = torch.from_numpy(self.embedding['pre_attention'])
        self.neighbors = trainer.graph.neighbors
        self.neighbor_mask = torch.from_numpy(self.neighbors > -1)
        neighbor_X = np.zeros((self.X.shape[0] * trainer.max_neighbors, self.X.shape[1]), dtype = np.float32)
        neighbor_X[self.neighbor_mask.reshape(-1)] = self.X[self.neighbors[self.neighbor_mask]]
        self.neighbor_X = torch.from_numpy(neighbor_X.reshape(self.X.shape[0], trainer.max_neighbors, self.X.shape[1]))#.to(self.device)
        self.distances = torch.from_numpy(trainer.graph.distances)#.to(self.device)
        self.batch_labels = torch.from_numpy(trainer.dataset.batch_labels)#.to(self.device)
        self.log_library_size = torch.from_numpy(trainer.dataset.log_library_size)#.to(self.device)

    
    @torch.no_grad()
    def _run_ablation_per_cell(
        self,
        ablation_mask: Optional[np.ndarray] = None,
        method: Literal['mask', 'mean', 'zero'] = 'mean',
    ):
        """
        Run a forward pass replacing neighbors with the mean embedding
        
        Parameters:
        -----------
        ablation_mask: Optional[np.ndarray]
            Boolean mask where True = ablate the cell's latent embedding
        """
        
        self.model.eval()
        
        x_hat = []
        if ablation_mask is None:
            ablation_mask = np.zeros(self.max_neighbors, dtype = bool)
        
        mean_z = self.mean_z.to(self.device)
        cell_z = self.X.to(self.device)
        neighbor_z = self.neighbor_X.clone().to(self.device)
        neighbor_mask = self.neighbor_mask.clone().to(self.device)
        distances = self.distances.to(self.device)
        batch_labels = self.batch_labels.to(self.device)
        log_library_size = self.log_library_size.to(self.device)
        ablation_tensor = torch.from_numpy(ablation_mask).to(self.device)
        if method == 'mean':
            neighbor_z[:,ablation_mask] = mean_z
        elif method == 'mask':
            neighbor_mask = torch.logical_and(neighbor_mask, ablation_tensor)
        elif method == 'zero':
            null_z = torch.zeros(mean_z.shape, dtype = torch.float32)
            neighbor_z[:,ablation_mask] = null_z.to(neighbor_z.device)
            
        # Mask cells with no neighbors from attention
        has_neighbors = neighbor_mask.any(dim = -1)
        post_attn_z = cell_z.clone()
        if has_neighbors.any():
            context = self.model.attention(
                central_z = cell_z[has_neighbors],
                neighbor_z = neighbor_z[has_neighbors],
                neighbor_mask = neighbor_mask[has_neighbors],
                distances = distances[has_neighbors],
            )
        
            post_attn_z[has_neighbors] += context
        
        return self.model.decoder(post_attn_z, log_library_size, batch_labels)[0].cpu().numpy()
    
    
    @torch.no_grad()
    def _run_single_permutation(
        self,
        perm: np.ndarray,
        method: Literal['mask', 'mean', 'zero'] = 'mean',
    ):
        """
        Run a forward pass replacing neighbors with the mean embedding
        
        Parameters:
        -----------
        ablation_mask: sample coalition order for neighbor ablation
        """
        
        scores = np.zeros((self.adata.shape[0], self.max_neighbors, self.adata.shape[1]), dtype = np.float32)
        self.model.eval()
        
        x_hat = []

        mean_z = self.mean_z.to(self.device)
        cell_z = self.X.to(self.device)
        neighbor_z = self.neighbor_X.clone().to(self.device)
        neighbor_mask = self.neighbor_mask.clone().to(self.device)
        distances = self.distances.to(self.device)
        batch_labels = self.batch_labels.to(self.device)
        log_library_size = self.log_library_size.to(self.device)
        
        # Mask cells with no neighbors from attention
        has_neighbors = neighbor_mask.any(dim = -1)
        post_attn_z = cell_z.clone()
        if has_neighbors.any():
            context = self.model.attention(
                central_z = cell_z[has_neighbors],
                neighbor_z = neighbor_z[has_neighbors],
                neighbor_mask = neighbor_mask[has_neighbors],
                distances = distances[has_neighbors],
            )
        
            post_attn_z[has_neighbors] += context
            
        prev = self.model.decoder(post_attn_z, log_library_size, batch_labels)[0].cpu().numpy()
        for p in perm:
            if method == 'mean':
                neighbor_z[:, p] = mean_z
            elif method == 'mask':
                neighbor_mask[:,p] = False
            elif method == 'zero':
                null_z = torch.zeros(mean_z.shape, dtype = torch.float32)
                neighbor_z[:,p] = null_z.to(neighbor_z.device)
                
            has_neighbors = neighbor_mask.any(dim = -1)
            post_attn_z = cell_z.clone()
        
            if has_neighbors.any():
                context = self.model.attention(
                    central_z = cell_z[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
            
                post_attn_z[has_neighbors] += context
            current = self.model.decoder(post_attn_z, log_library_size, batch_labels)[0].cpu().numpy()
            scores[:,p] += prev - current
            prev = current
        return scores
    
    
    def run_mc_per_cell(
        self,
        n_iter: int = 100,
        method: Literal['mask', 'mean', 'zero'] = 'mean',
        random_state: int = 42
    ):
        rng = np.random.default_rng(random_state)
        
        scores = np.zeros([self.adata.shape[0], self.max_neighbors, self.adata.shape[1]], dtype = np.float32)
                
        for i in tqdm(range(n_iter), desc = "Calculating MC Shap score per cell"):
            coalition = rng.permutation(self.max_neighbors).tolist()
            scores += self._run_single_permutation(coalition, method = method)
        
        scores = scores / n_iter
        scores[~self.neighbor_mask.cpu().numpy(),:] = 0
        
        return scores
    
    def get_ablated_embedding(
        self,
        method: Literal['mean', 'zero'] = 'zero'
    ):
        
        self.model.eval()
        
        out = []
        
        for batch in self.dataloader:
            if method == 'mean':
                cell_z = mean_z.to(self.device)
            elif method == 'zero':
                cell_z = torch.zeros(self.mean_z.shape, dtype = torch.float32).to(self.device)

            neighbor_z = batch['neighbor_X'].clone().to(self.device)
            neighbor_mask = batch['neighbor_mask'].clone().to(self.device)
            distances = batch['distances'].to(self.device)
            cell_idx = batch['cell_idx'].to(self.device)
            log_library_size = batch['log_library_size'].to(self.device)
            batch_neighbors = torch.from_numpy(self.neighbors[cell_idx.cpu().numpy()]).to(self.device)
            
            # Mask cells with no neighbors from attention
            has_neighbors = neighbor_mask.any(dim = -1)
            post_attn_z = cell_z.clone()
            
            
            if has_neighbors.any():
                context = self.model.attention(
                    central_z = cell_z[has_neighbors],
                    neighbor_z = neighbor_z[has_neighbors],
                    neighbor_mask = neighbor_mask[has_neighbors],
                    distances = distances[has_neighbors],
                )
                
                post_attn_z[has_neighbors] += context
            
            ou.append(post_attn_z.cpu().numpy())
        
        out = np.concatenate(out, axis = 0)
        
        return out