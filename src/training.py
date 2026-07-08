from dataset import *
from neighbors import *
from autoencoder import *
from utils import *
from EarlyStopping import *
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time

class SpatialAutoencoderTrainer:
    
    def __init__(
        self,
        graph: SpatialNeighbors,

        # Dataset kwargs:
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        
        # Dataloader kwargs:
        batch_size: int = 2**10,
        num_workers: int = 0,
        
        # Autoencoder kwargs
        latent_dim: int = 10,
        hidden_dims: List[int] = [],
        num_heads: int = 1,
        dropout: float = 0.1,
        batch_dim: int = 5,
        rbf_n_basis: int = 8,
        rbf_spacing: Literal['linear', 'log'] = 'linear',
        activation: Literal['gelu', 'relu', 'leaky_relu'] = 'gelu',
        
        # Optimizer kwargs
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_patience: float = 20,
        lr_delta: float = 1e-4,
        lr_factor: float = 0.5,
        min_lr: float = 1e-5,
        
        # Early stopping kwargs
        early_stop_patience: int = 20,
        early_stop_delta: float = 1e-4,
        early_stop_offset: int = 25,
        
        # Loss kwargs
        loss_fn: Literal['mse', 'huber', 'zinb'] = 'zinb',
        delta: float = 1.0,
        beta_kl_max: float = 5,
        beta_ramp_start: int = 25,
        beta_ramp_end: int = 50,
        # Post attention embedding = gamma * attention_context + alpha * pre_attention_embedding
        # Anneal alpha to get spatial context first
        # Anneal gamma to train without spatial context first
        alpha_ramp_start: int = 0,
        alpha_ramp_end: int = 0,
        alpha_max: float = 1,
        gamma_ramp_start: int = 50,
        gamma_ramp_end: int = 100,
        gamma_max: float = 1,
        mask_ramp_start: int = 0,
        mask_ramp_end: int = 0,
        mask_pct_max: float = 0,
        
        # Training kwargs
        max_epochs: int = 1000,
        grad_clip_norm: float = 1.0,
        log_every: int = 20,
        
        # Other kwargs:
        device: str = 'auto',
        dst_dir: Optional[str] = None,
        random_state: int = 42
    ):
        self.graph = graph
        
        self.random_state = random_state
        self.batch_key = batch_key
        self.layer = layer

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.rbf_n_basis = rbf_n_basis
        self.rbf_spacing = rbf_spacing
        self.activation = activation
        
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        # Setup batches
        self.n_batches = 1
        self.batch_dim = 0
        self.labels = np.zeros(len(self.graph.adata), dtype = np.int32)
        if self.batch_key:
            if self.batch_key not in self.graph.adata.obs:
                raise ValueError(f"Batch key {self.batch_key} not in adata.obs")
            labels, unique = pd.factorize(self.graph.adata.obs[self.batch_key])
            self.labels = labels
            self.n_batches = len(unique)
            self.graph.adata.obs[f'{self.batch_key}_int'] = labels
        self.learning_rate = learning_rate
        self.weigh_decay = weight_decay
        self.lr_patience = lr_patience
        self.lr_delta = lr_delta
        self.lr_factor = lr_factor
        self.min_lr = min_lr
                
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta   
        self.early_stop_offset = early_stop_offset    
        
        self.loss_fn = loss_fn
        self.delta = delta
        self.beta_kl_max = beta_kl_max
        self.beta_ramp_start = beta_ramp_start
        self.beta_ramp_end = beta_ramp_end
        self.alpha_max = alpha_max
        self.alpha_ramp_start = alpha_ramp_start
        self.alpha_ramp_end = alpha_ramp_end
        self.gamma_max = gamma_max
        self.gamma_ramp_start = gamma_ramp_start
        self.gamma_ramp_end = gamma_ramp_end
        self.mask_ramp_start = mask_ramp_start
        self.mask_ramp_end = mask_ramp_end
        self.mask_pct_max = mask_pct_max
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        
        self.dst_dir = dst_dir
        if dst_dir:
            if not os.path.isdir(dst_dir):
                Path(dst_dir).mkdir(parents = True, exist_ok = True)
        
        # Initialize setup variables
        self.dataset = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.loss = None
        self.embedding = None
        
    @property
    def adata(self):
        return self.graph.adata
    
    @adata.setter
    def adata(self, value):
        self.graph.adata = value

    def setup(self, verbose: bool = True):

        if verbose:
            print("Setting up spatial datasets and dataloaders")
        self._setup_dataset()
        self._setup_dataloaders()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        if verbose:
            print("--- Done with setup ---")

        
    def _setup_dataset(self):
        if self.graph is None:
            self._setup_spatial_graph()
            
        self.dataset = SpatialDataset.from_graph(
            self.graph,
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
        )
        
        self.train_dataset = SpatialDataset.from_graph(
            self.graph,
            cell_indices = self.graph.split_idx['train_idx'],
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
        )
        
        self.val_dataset = SpatialDataset.from_graph(
            self.graph,
            cell_indices = self.graph.split_idx['val_idx'],
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
        )
        
        self.test_dataset = SpatialDataset.from_graph(
            self.graph,
            cell_indices = self.graph.split_idx['test_idx'],
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
        )
        
        self.X = self.adata.X if self.layer is None else self.adata.layers[self.layer]
        if issparse(self.X):
            self.X = self.X.toarray()
        self.X = torch.from_numpy(self.X)
            
        
    def _setup_dataloaders(self):
        if self.dataset is None:
            self._setup_dataset()

        def make_stratified_sampler(idxs):
            counts = np.bincount(self.labels[idxs])
            weights = 1.0 / counts[self.labels[idxs]]
            g = torch.Generator()
            g.manual_seed(self.random_state)

            return WeightedRandomSampler(
                weights = torch.tensor(weights, dtype = torch.float32),
                num_samples = len(idxs),
                replacement = True,
                generator = g
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler = make_stratified_sampler(self.graph.split_idx['train_idx']),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            collate_fn = lambda x: x,
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            drop_last = False,
            shuffle = False,
            collate_fn = lambda x: x,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            drop_last = False,
            shuffle = False,
            collate_fn = lambda x: x,
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = lambda x: x,
        )
        
    def _setup_model(self):
        if self.graph is None:
            self._setup_spatial_graph()
        
        self.model = SpatialAutoEncoder(
            n_genes = self.adata.shape[1],
            latent_dim = self.latent_dim,
            hidden_dims = self.hidden_dims,
            num_heads = self.num_heads,
            dropout = self.dropout,
            n_batches = self.n_batches,
            batch_dim = self.batch_dim,
            d_min  = self.adata.obsp[self.graph.distance_key].data.min(),
            d_max = self.adata.obsp[self.graph.distance_key].data.max(),
            rbf_n_basis = self.rbf_n_basis,
            rbf_spacing = self.rbf_spacing,
            activation = self.activation,
        ).to(self.device)
        
        
    def _setup_optimizer(self):
        if self.model is None:
            self._setup_model()
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weigh_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = 'min',
            factor = self.lr_factor,
            patience = self.lr_patience,
            threshold = self.lr_delta,
            min_lr = self.min_lr
        )
        
        self.early_stopping = EarlyStopping(patience = self.early_stop_patience, delta = self.early_stop_delta)
        
    def _setup_loss(self):

        self.loss = ReconstructionLoss(
            loss_fn = self.loss_fn,
            delta = self.delta,
        ).to(self.device)
        
        
    def forward_batch(self, batch, alpha = 1, gamma = 1, mask_pct = 0.0, embedding_only = False, verbose = False):
            
        cell_idx = batch['cell_idx']
        neighbor_idx = batch['neighbor_idx']
        neighbor_mask = batch['neighbor_mask']
        distances = batch['distances'].to(self.device)
        log_library_size = batch['log_library_size'].to(self.device)
        batch_label = batch['batch_label'].to(self.device)

        # Get all unique indices (central + neighbors), keep on cpu since full expression matrix is on cpu
        neighbor_idx_flat = neighbor_idx[neighbor_mask]
        all_needed_idx = torch.cat([cell_idx, neighbor_idx_flat], dim = 0)
        unique_idx, inverse = torch.unique(all_needed_idx, sorted = True, return_inverse = True)

        # Get expression for all unique cells in batch and move to gpu
        cell_idx = cell_idx.to(self.device)
        neighbor_idx = neighbor_idx.to(self.device)
        neighbor_mask = neighbor_mask.to(self.device)
        unique_X = self.X[unique_idx].to(self.device)

        # Forward pass to encode all Z
        unique_z = self.model.encoder(unique_X)

        # Get central embeddings
        central_inverse = inverse[:cell_idx.shape[0]]
        cell_z = unique_z[central_inverse]
        
        # Get neighbor embeddings
        neighbor_inverse = inverse[cell_idx.shape[0]:]
        neighbor_z = torch.zeros(
            (cell_idx.shape[0], neighbor_idx.shape[1], cell_z.shape[1]),
            device = self.device,
            dtype = cell_z.dtype
        )
        neighbor_z[neighbor_mask] = unique_z[neighbor_inverse]
        
        # post_attention_z = alpna * mask * pre_attention_z + gamma * sigmoid(delta) * context
        has_neighbors = neighbor_mask.any(dim = -1)
        post_attn_z = cell_z.clone()
        
        weights = None
        if has_neighbors.any():
            context, weights = self.model.attention(
                central_z = cell_z[has_neighbors],
                neighbor_z = neighbor_z[has_neighbors],
                neighbor_mask = neighbor_mask[has_neighbors],
                distances = distances[has_neighbors],
            )
            
            # Randomly mask pre-attention embeddings to force spatial awareness
            mask = torch.bernoulli(
                torch.full((cell_z.shape[0], 1), 1 - mask_pct, device=cell_z.device)
            )
            post_attn_z[has_neighbors] *= mask[has_neighbors]
            
            post_attn_z[has_neighbors] *= alpha
            post_attn_z[has_neighbors] += gamma * torch.sigmoid(self.model.delta) * context

        mu = self.model.mu(post_attn_z)
        log_var = self.model.log_var(post_attn_z)
        
        mu_z = self.model.reparameterize(mu, log_var)
        
        if embedding_only:
            return mu_z, cell_z, post_attn_z

        mu_x, theta, pi = self.model.decode(mu_z, log_library_size, batch_label)
        
        return {
            'mu_z': mu_z,
            'log_var': log_var,
            'mu_x': mu_x,
            'theta': theta,
            'pi': pi,
            'pre_attn_z': cell_z,
            'post_attn_z': post_attn_z,
            'cell_X': unique_X[central_inverse],
            'attn_weights': weights
        }
        
    def train(self, random_state: int = 42, verbose: bool = True):
        
        set_random_seed(random_state, self.device)
        history = []
        for epoch in (pbar := tqdm(range(self.max_epochs), desc = 'Training model', disable = not verbose)):
            
            # Training loop
            n_train = 0
            train_loss = 0.0    
            train_recon = 0.0
            train_kl = 0.0   
            beta_kl = get_anneal_ramp_param(epoch, self.beta_ramp_start, self.beta_ramp_end, self.beta_kl_max, method = 'cosine')
            alpha = get_anneal_ramp_param(epoch, self.alpha_ramp_start, self.alpha_ramp_end, self.alpha_max, method = 'cosine')
            gamma = get_anneal_ramp_param(epoch, self.gamma_ramp_start, self.gamma_ramp_end, self.gamma_max, method = 'cosine')
            mask_pct = get_anneal_ramp_param(epoch, self.mask_ramp_start, self.mask_ramp_end, self.mask_pct_max, method = 'cosine')
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                # Update progress bar description in 20% batch increments
                if (max(i, 1) % max(1, len(self.train_loader) // 5)) == 0 or (i == len(self.train_loader) - 1):
                    pbar.set_description(f'Running training loop: batch = {i+1}/{len(self.train_loader)} (Early stop count = {self.early_stopping.counter})')
                self.optimizer.zero_grad()
                outputs = self.forward_batch(batch, alpha = alpha, gamma = gamma, mask_pct = mask_pct)
                loss, recon_loss, kl_loss = self.loss(
                    outputs['mu_z'],
                    outputs['log_var'],
                    outputs['mu_x'],
                    outputs['theta'],
                    outputs['pi'],
                    outputs['cell_X'],
                    beta_kl,
                )
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm = self.grad_clip_norm
                )
                
                self.optimizer.step()
                n = batch['cell_idx'].shape[0]
                n_train += n
                train_loss += n * loss.item()
                train_recon += n * recon_loss
                train_kl += n * kl_loss

            train_loss /= n_train
            train_recon /= n_train
            train_kl /= n_train
            
            # Validation loop
            n_val = 0
            val_loss = 0.0
            val_recon = 0.0
            val_kl = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    if (max(1, i) % max(1, len(self.val_loader)) // 2) == 0 or (i == len(self.val_loader) - 1):
                        pbar.set_description(f'Running validation loop: batch = {i+1}/{len(self.val_loader)} (Early stop count = {self.early_stopping.counter})')
                    outputs = self.forward_batch(batch, alpha = alpha, gamma = gamma, verbose = False)
                    loss, recon_loss, kl_loss = self.loss(
                        outputs['mu_z'],
                        outputs['log_var'],
                        outputs['mu_x'],
                        outputs['theta'],
                        outputs['pi'],
                        outputs['cell_X'],
                        beta_kl
                    )

                    n = batch['cell_idx'].shape[0]
                    n_val += n
                    val_loss += loss.item() * n
                    val_recon += recon_loss * n
                    val_kl += kl_loss * n

                val_loss /= n_val
                val_recon /= n_val
                val_kl /= n_val
            
            if epoch > self.beta_ramp_start:
                self.scheduler.step(val_loss)
            if epoch > max(self.alpha_ramp_end, self.beta_ramp_end, self.gamma_ramp_end, self.mask_ramp_end) + self.early_stop_offset:
                self.early_stopping(val_loss, self.model)
                
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_recon': train_recon,
                'train_kl': train_kl,
                'val_loss': val_loss,
                'val_recon': val_recon,
                'val_kl': val_kl,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'early_stopping': self.early_stopping.counter,
                'beta_kl': beta_kl,
                'alpha': alpha,
                'gamma': gamma,
                'delta': torch.sigmoid(self.model.delta).detach().cpu().numpy().mean(),
                'mask_pct': mask_pct
            })
            if self.early_stopping.early_stop:
                if verbose:
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                break
        self.early_stopping.load_best_model(self.model)
        self.history = pd.DataFrame(history)

    def plot_history(self, show: bool = True):
        """
        Plot the loss per epoch for each loss term for test and validation data
        Show the lambda_mmd ramp and learning rate decreases below the loss plot
        """
        epochs = self.history['epoch'].values
        train_r = self.history['train_recon'].values
        train_l = self.history['train_loss'].values
        train_k = self.history['train_kl'].values
        val_r = self.history['val_recon'].values
        val_l = self.history['val_loss'].values
        val_k = self.history['val_kl'].values
        lr = self.history['learning_rate'].values
        beta_kl = self.history['beta_kl'].values
        alpha = self.history['alpha'].values
        gamma = self.history['gamma'].values
        delta = self.history['delta'].values
        mask_pct = self.history['mask_pct'].values
        
        fig, ax = plt.subplots(2, 1, figsize = (6, 5), height_ratios = [3, 1])
        
        ax[0].plot(epochs, train_l, 'k-', label = 'Train loss')
        ax[0].plot(epochs, val_l, 'k--', label = 'Validation loss')
        ax[0].plot(epochs, train_r, 'C0-', label = 'Train Reconstruction loss')
        ax[0].plot(epochs, val_r, 'C0--', label = 'Validation Reconstruction loss')
        ax[0].plot(epochs, train_k * beta_kl, 'C2-', label = 'Train KL loss')
        ax[0].plot(epochs, val_k * beta_kl, 'C2--', label = 'Validation KL loss')

        ax[0].set_ylabel("Loss")

        ax[0].legend(loc = 'center left', bbox_to_anchor = [1.15, 0.5])
        ax[0].grid()
        
        ax[1].plot(epochs, lr, 'k-', alpha = 0.7, label = 'Learning Rate')
        ax[1].set_ylabel("Learning Rate")
        ax1_1 = ax[1].twinx()
        ax1_1.plot(epochs, (beta_kl) / (beta_kl.max() if beta_kl.max() > 0 else 1), 'C2-', alpha = 0.7, label = r'$\beta_\text{KL}$' + f' [max = {beta_kl.max():.3f}]')
        ax1_1.plot(epochs, (alpha) / (alpha.max() if alpha.max() > 0 else 1), 'm-', alpha = 0.7, label = r'$\alpha$' + f' [max = {alpha.max():.3f}]')
        ax1_1.plot(epochs, (gamma) / (gamma.max() if gamma.max() > 0 else 1), 'C3-', alpha = 0.7, label = r'$\gamma$' + f' [max = {gamma.max():.3f}]')
        ax1_1.plot(epochs, delta, 'C4', alpha = 0.7, label = r'sigmoid$(\delta)$')
        ax1_1.plot(epochs, mask_pct, 'C1-', alpha = 0.7, label = 'Mask Pct')
        ax1_1.set_ylabel("Other params [a.u.]")
        ax[1].grid()
        lines, labels = ax[1].get_legend_handles_labels()
        lines2, labels2 = ax1_1.get_legend_handles_labels()
        ax1_1.legend(lines + lines2, labels + labels2, loc = 'center left', bbox_to_anchor = [1.15, 0.5])

        if show:
            plt.show()
            return
        return fig
    
    @torch.no_grad()
    def eval_test(self):
        n_test = 0
        test_loss = 0.0
        test_recon = 0.0
        test_kl = 0.0
        self.model.eval()
        for batch in tqdm(self.test_loader, desc = "Evaluating on test dataset"):
            outputs = self.forward_batch(batch)
            loss, recon_loss, kl_loss = self.loss(
                outputs['mu_z'],
                outputs['log_var'],
                outputs['mu_x'],
                outputs['theta'],
                outputs['pi'],
                outputs['cell_X'],
                self.max_beta_kl
            )
            
            n = batch['cell_idx'].shape[0]
            n_test += n
            test_loss += n * loss.item()
            test_recon += n * recon_loss
            test_kl += n * kl_loss
        test_loss /= n_test
        test_recon /= n_test
        test_kl /= n_test
        return test_loss, test_recon, test_kl
    
    @torch.no_grad()
    def get_embedding(self, alpha = 1.0, gamma = 1.0):
        self.model.eval()
        embeddings = {'z': [], 'pre_attention': [], 'post_attention': []}
        cell_indices = []
        
        for batch in self.dataloader:
            z, pre_z, post_z = self.forward_batch(batch, embedding_only = True, alpha = alpha, gamma = gamma)
            embeddings['z'].append(z.cpu().numpy())
            embeddings['pre_attention'].append(pre_z.cpu().numpy())
            embeddings['post_attention'].append(post_z.cpu().numpy())
            cell_indices.append(batch['cell_idx'])
            
        self.embedding = {
            'z': np.concatenate(embeddings['z'], axis = 0),
            'pre_attention': np.concatenate(embeddings['pre_attention'], axis = 0),
            'post_attention': np.concatenate(embeddings['post_attention'], axis = 0),
            'cell_idx': np.concatenate(cell_indices, axis = 0)
        }
        
        return self.embedding
    
    @torch.no_grad()
    def predict(self, alpha = 1.0, gamma = 1.0):
        self.model.eval()
        x_hat = []
        pi = []
        t1 = time.perf_counter()
        for batch in self.dataloader:
            result = self.forward_batch(batch, alpha = alpha, gamma = gamma)
            x_hat.append(result['mu_x'].cpu().numpy())
            pi.append(result['pi'].cpu().numpy())
            
        x_hat = np.concatenate(x_hat, axis = 0)
        pi = np.concatenate(pi, axis = 0)
        theta = torch.exp(self.model.decoder.log_theta).detach().cpu().numpy()
        return {
            'mu': x_hat,
            'pi': pi,
            'theta': theta
        }

    def save(
        self,
        dst_dir: str = '',
    ):
        if not os.path.isdir(dst_dir):
            Path(dst_dir).mkdir(exist_ok = True, parents = True)
        if self.history is not None:
            self.history.to_csv(f'{dst_dir}/history.csv')
        self.model.save(dst_dir)
        self.adata.write_h5ad(f'{dst_dir}/adata.h5ad')