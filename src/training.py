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
from collections import defaultdict

class SpatialAutoencoderTrainer:
    
    def __init__(
        self,
        graph: SpatialNeighbors,

        # Dataset kwargs:
        batch_key: Optional[str] = None,
        celltype_key: Optional[str] = None,
        layer: Optional[str] = None,
        
        # Dataloader kwargs:
        batch_size: int = 2**10,
        num_workers: int = 0,
        
        # Autoencoder kwargs
        latent_dim: int = 10,
        hidden_dims: List[int] = [],
        attn_dim: Optional[int] = None,
        topk: int = 20,
        project_inputs: bool = True,
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
        beta_kl_max: float = 1,
        beta_ramp_start: int = 25,
        beta_ramp_end: int = 75,
        alpha_ramp_start: int = 100,
        alpha_ramp_end: int = 150,
        alpha_max: float = 1,
        gamma_ramp_start: int = 175,
        gamma_ramp_end: int = 225,
        gamma: float = 0.5,
        lambda_intrinsic: float = 0.75,
        lambda_mu_prior: float = 1e-3,
        lambda_delta: float = 1e-3,
        
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
        self.celltype_key = celltype_key
        self.layer = layer

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.attn_dim = attn_dim if attn_dim is not None else latent_dim
        self.topk = topk
        self.project_inputs = project_inputs
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
        
        self.n_celltypes = 1
        if self.celltype_key:
            if self.celltype_key not in self.graph.adata.obs:
                raise ValueError(f"Celltype key {self.celltype_key} not in adata.obs")
            labels, unique = pd.factorize(self.graph.adata.obs[self.celltype_key])
            self.celltype_labels = labels
            self.n_celltypes = len(unique)
            self.graph.adata.obs[f'{self.celltype_key}_int'] = labels
        self.learning_rate = learning_rate
        self.weigh_decay = weight_decay
        self.lr_patience = lr_patience
        self.lr_delta = lr_delta
        self.lr_factor = lr_factor
        self.min_lr = min_lr
                
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta   
        self.early_stop_offset = early_stop_offset    
        
        self.beta_kl_max = beta_kl_max
        self.beta_ramp_start = beta_ramp_start
        self.beta_ramp_end = beta_ramp_end
        self.alpha_max = alpha_max
        self.alpha_ramp_start = alpha_ramp_start
        self.alpha_ramp_end = alpha_ramp_end
        self.gamma_ramp_start = gamma_ramp_start
        self.gamma_ramp_end = gamma_ramp_end
        self.gamma = gamma
        self.lambda_intrinsic = lambda_intrinsic
        self.lambda_mu_prior = lambda_mu_prior
        self.lambda_delta = lambda_delta
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
            celltype_key = f'{self.celltype_key}_int' if self.celltype_key else None,
        )
        
        self.train_dataset = SpatialDataset.from_graph(
            self.graph,
            cell_indices = self.graph.split_idx['train_idx'],
            celltype_key = f'{self.celltype_key}_int' if self.celltype_key else None,
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
        )
        
        self.val_dataset = SpatialDataset.from_graph(
            self.graph,
            cell_indices = self.graph.split_idx['val_idx'],
            celltype_key = f'{self.celltype_key}_int' if self.celltype_key else None,
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
        )
        
        self.test_dataset = SpatialDataset.from_graph(
            self.graph,
            cell_indices = self.graph.split_idx['test_idx'],
            celltype_key = f'{self.celltype_key}_int' if self.celltype_key else None,
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
            attn_dim = self.attn_dim,
            topk = self.topk,
            project_inputs = self.project_inputs,
            num_heads = self.num_heads,
            dropout = self.dropout,
            n_celltypes = self.n_celltypes,
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

        self.loss = ReconstructionLoss().to(self.device)
        
        
    def forward_batch(self, batch, alpha = 1.0, embedding_only = False, verbose = False):
            
        cell_idx = batch['cell_idx']
        celltype_labels = batch['celltype_label'].to(self.device)
        neighbor_idx = batch['neighbor_idx']
        neighbor_mask = batch['neighbor_mask']
        distances = batch['distances'].to(self.device)
        log_library_size = batch['log_library_size'].to(self.device)
        batch_label = batch['batch_label'].to(self.device)

        mu_z_prior = self.model.celltype_prior(celltype_labels)
        
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
        unique_mu_z, unique_log_var_z = self.model.encoder(unique_X)

        # Get central embeddings
        central_inverse = inverse[:cell_idx.shape[0]]
        z_intrinsic = self.model.reparameterize(unique_mu_z[central_inverse], unique_log_var_z[central_inverse])
        
        # Get neighbor embeddings
        neighbor_inverse = inverse[cell_idx.shape[0]:]
        neighbor_z = torch.zeros(
            (cell_idx.shape[0], neighbor_idx.shape[1], z_intrinsic.shape[1]),
            device = self.device,
            dtype = z_intrinsic.dtype
        )
        neighbor_z[neighbor_mask] = unique_mu_z[neighbor_inverse].detach()
        
        has_neighbors = neighbor_mask.any(dim = -1)
        z_spatial = torch.zeros((z_intrinsic.shape[0], self.attn_dim), dtype = torch.float32, device = z_intrinsic.device)
        
        weights = None
        if has_neighbors.any():
            query = z_intrinsic.detach()
            context, weights = self.model.attention(
                central_z = query[has_neighbors],
                neighbor_z = neighbor_z[has_neighbors],
                neighbor_mask = neighbor_mask[has_neighbors],
                distances = distances[has_neighbors],
            )
            z_spatial[has_neighbors] = context
        
        if embedding_only:
            return z_intrinsic, z_spatial

        mu, theta, pi, mu_intrinsic, delta = self.model.decode(z_intrinsic, z_spatial, log_library_size, batch_label, alpha)
        
        return {
            'mu_z_prior': mu_z_prior,
            'mu_z': z_intrinsic,
            'log_var': unique_log_var_z[central_inverse],
            'mu' : mu,
            'theta': theta,
            'pi': pi,
            'mu_intrinsic': mu_intrinsic,
            'delta': delta,
            'z_spatial': z_spatial,
            'attn_weights': weights,
            'cell_X': unique_X[central_inverse],
        }
        
    def _calc_loss_from_outputs(self, outputs, beta_kl = 1.0, gamma = 1.0, lambda_mu_prior = 1e-3, lambda_delta = 1e-3):
        loss = self.loss(
            outputs['mu_z_prior'],
            outputs['mu_z'],
            outputs['log_var'],
            outputs['mu'],
            outputs['mu_intrinsic'],
            outputs['delta'],
            outputs['pi'],
            outputs['theta'],
            outputs['cell_X'],
            beta_kl,
            gamma,
            lambda_mu_prior,
            lambda_delta
        )
        
        return loss
        
    def train(self, random_state: int = 42, verbose: bool = True):
        
        set_random_seed(random_state, self.device)
        history = []
        for epoch in (pbar := tqdm(range(self.max_epochs), desc = 'Training model', disable = not verbose)):
            
            # Training loop
            n_train = 0
            train_loss = defaultdict(float) 
            beta_kl = get_anneal_ramp_param(epoch, self.beta_ramp_start, self.beta_ramp_end, self.beta_kl_max, method = 'cosine')
            alpha = get_anneal_ramp_param(epoch, self.alpha_ramp_start, self.alpha_ramp_end, self.alpha_max, method = 'cosine')
            gamma = get_anneal_ramp_param(epoch, self.gamma_ramp_start, self.gamma_ramp_end, min_param = 1.0, max_param = self.gamma, method = 'cosine')
            lambda_delta = self.lambda_delta * max(1e-4, alpha)
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                # Update progress bar description in 20% batch increments
                if (max(i, 1) % max(1, len(self.train_loader) // 5)) == 0 or (i == len(self.train_loader) - 1):
                    pbar.set_description(f'Running training loop: batch = {i+1}/{len(self.train_loader)} (Early stop count = {self.early_stopping.counter})')
                self.optimizer.zero_grad()
                outputs = self.forward_batch(batch)
                loss= self._calc_loss_from_outputs(outputs, beta_kl, gamma, self.lambda_mu_prior, lambda_delta)
                    
                loss['loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm = self.grad_clip_norm
                )
                
                self.optimizer.step()
                n = batch['cell_idx'].shape[0]
                n_train += n
                for l in loss:
                    if l == 'loss':
                        train_loss[l] += n * loss[l].item()
                    else:
                        train_loss[l] += n * loss[l]

            for l in train_loss:
                train_loss[l] /= n_train
            
            # Validation loop
            n_val = 0
            val_loss = defaultdict(float)

            self.model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    if (max(1, i) % max(1, len(self.val_loader)) // 2) == 0 or (i == len(self.val_loader) - 1):
                        pbar.set_description(f'Running validation loop: batch = {i+1}/{len(self.val_loader)} (Early stop count = {self.early_stopping.counter})')
                    outputs = self.forward_batch(batch, verbose = False)
                    loss = self._calc_loss_from_outputs(outputs, beta_kl, gamma, self.lambda_mu_prior, lambda_delta)

                    n = batch['cell_idx'].shape[0]
                    n_val += n

                    for l in loss:
                        if l == 'loss':
                            val_loss[l] += n * loss[l].item()
                        else:
                            val_loss[l] += n * loss[l]

                for l in val_loss:
                    val_loss[l] /= n_val
                
            if epoch > self.beta_ramp_start:
                self.scheduler.step(val_loss['loss'])
            if epoch > max(self.beta_ramp_end, self.gamma_ramp_end, self.alpha_ramp_end) + self.early_stop_offset:
                self.early_stopping(val_loss['loss'], self.model)
                
            history.append({
                'epoch': epoch,
                'train_loss': train_loss['loss'],
                'train_recon': train_loss['loss_recon'],
                'train_recon_intrinsic': train_loss['loss_recon_intrinsic'],
                'train_kl': train_loss['loss_kl'],
                'train_prior_reg': train_loss['loss_prior_reg'],
                'train_delta_reg': train_loss['loss_delta_reg'],
                'val_loss': val_loss['loss'],
                'val_recon': val_loss['loss_recon'],
                'val_recon_intrinsic': val_loss['loss_recon_intrinsic'],
                'val_kl': val_loss['loss_kl'],
                'val_prior_reg': val_loss['loss_prior_reg'],
                'val_delta_reg': val_loss['loss_delta_reg'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'early_stopping': self.early_stopping.counter,
                'beta_kl': beta_kl,
                'alpha': alpha,
                'gamma': gamma,
                'lambda_delta': lambda_delta,
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
        train_r_i = self.history['train_recon_intrinsic'].values
        train_p_r = self.history['train_prior_reg'].values
        train_d_r = self.history['train_delta_reg'].values
        train_l = self.history['train_loss'].values
        train_k = self.history['train_kl'].values
        val_r = self.history['val_recon'].values
        val_r_i = self.history['val_recon_intrinsic'].values
        val_p_r = self.history['val_prior_reg'].values
        val_d_r = self.history['val_delta_reg'].values
        val_l = self.history['val_loss'].values
        val_k = self.history['val_kl'].values
        lr = self.history['learning_rate'].values
        beta_kl = self.history['beta_kl'].values
        alpha = self.history['alpha'].values
        gamma = self.history['gamma'].values
        lambda_delta = self.history['lambda_delta'].values
        
        fig, ax = plt.subplots(2, 1, figsize = (6, 5), height_ratios = [3, 1])
        
        ax[0].plot(epochs, train_l, 'k-', label = r'$\mathcal{L}_\text{Train}$')
        ax[0].plot(epochs, val_l, 'k--', label = r'$\mathcal{L}_\text{Val}$')
        ax[0].plot(epochs, train_r, 'C0-', label = r'$\mathcal{L}_\text{Train}(\text{ZINB})$')
        ax[0].plot(epochs, val_r, 'C0--', label = r'$\mathcal{L}_\text{Val}(\text{ZINB})$')
        ax[0].plot(epochs, gamma * train_r_i, 'C1-', label = r'$\gamma*\mathcal{L}_\text{Train}(\text{ZINB}_\text{intr})$')
        ax[0].plot(epochs, gamma * val_r_i, 'C1--', label = r'$\gamma*\mathcal{L}_\text{Val}(\text{ZINB}_\text{intr})$')
        ax[0].plot(epochs, beta_kl * train_k, 'C2-', label = r'$\beta_\text{KL}*\mathcal{L}_\text{Train}(\text{KL})$')
        ax[0].plot(epochs, beta_kl * val_k, 'C2--', label = r'$\beta_\text{KL}*\mathcal{L}_\text{Val}(\text{KL})$')
        ax[0].plot(epochs, self.lambda_mu_prior * train_p_r, 'C3-', label = r'$\lambda_{z_\mu}*||\mu_{z,i}||_\text{Train}$')
        ax[0].plot(epochs, self.lambda_mu_prior * val_p_r, 'C3--', label = r'$\lambda_{z_\mu}*||\mu_{z,i}||_\text{Val}$')
        ax[0].plot(epochs, lambda_delta * train_d_r, 'C4-', label = r'$\lambda_\Delta*||\Delta||_\text{Train}$')
        ax[0].plot(epochs, lambda_delta * val_d_r, 'C4--', label = r'$\lambda_\Delta*||\Delta||_\text{Val}$')
        ax[0].set_ylabel("Loss")
        ax[0].set_yscale("symlog", linthresh = 1e-3)
        ax[0].legend(loc = 'center left', bbox_to_anchor = [1.15, 0.5])
        ax[0].grid()
        
        ax[1].plot(epochs, lr, 'k-', alpha = 0.7, label = 'Learning Rate')
        ax[1].set_ylabel("Learning Rate")
        ax1_1 = ax[1].twinx()
        ax1_1.plot(epochs, (beta_kl) / (beta_kl.max() if beta_kl.max() > 0 else 1), 'C2--', alpha = 0.7, label = r'$\beta_\text{KL}$' + f' [max = {beta_kl.max():.3f}]')
        ax1_1.plot(epochs, (alpha) / (alpha.max() if alpha.max() > 0 else 1), 'C0--', alpha = 0.7, label = r'$\alpha$' + f' [max = {alpha.max():.3f}]')
        ax1_1.plot(epochs, gamma, 'C1--', alpha = 0.7, label = r'$\gamma$')
        ax1_1.plot(epochs, (lambda_delta) / (lambda_delta.max() if lambda_delta.max() > 0 else 1), 'C4--', alpha = 0.7, label = r'$\lambda_\Delta$' + f' [max = {lambda_delta.max()}]')
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
        self.model.eval()
        for batch in tqdm(self.test_loader, desc = "Evaluating on test dataset"):
            outputs = self.forward_batch(batch)
            loss, recon_loss, kl_loss = self._calc_loss_from_outputs(
                outputs,
                self.max_beta_kl,
                self.gamma,
                self.lambda_mu_prior,
                self.lambda_delta
            )
            
            n = batch['cell_idx'].shape[0]
            n_test += n
            test_loss += n * loss['loss'].item()

        test_loss /= n_test

        return test_loss
    
    @torch.no_grad()
    def get_embedding(self, alpha = 1.0, gamma = 1.0):
        self.model.eval()
        embeddings = {'z_intrinsic': [], 'z_spatial': []}
        cell_indices = []
        
        for batch in self.dataloader:
            z_intrinsic, z_spatial = self.forward_batch(batch, embedding_only = True)
            embeddings['z_intrinsic'].append(z_intrinsic.cpu().numpy())
            embeddings['z_spatial'].append(z_spatial.cpu().numpy())
            cell_indices.append(batch['cell_idx'])
            
        self.embedding = {
            'z_intrinsic': np.concatenate(embeddings['z_intrinsic'], axis = 0),
            'z_spatial': np.concatenate(embeddings['z_spatial'], axis = 0),
            'cell_idx': np.concatenate(cell_indices, axis = 0)
        }
        
        return self.embedding
    
    @torch.no_grad()
    def predict(self):
        self.model.eval()
        mu = []
        pi = []
        mu_intrinsic = []
        delta = []
        for batch in self.dataloader:
            output = self.forward_batch(batch)
            mu.append(output['mu'].cpu().numpy())
            pi.append(output['pi'].cpu().numpy())
            mu_intrinsic.append(output['mu_intrinsic'].cpu().numpy())
            delta.append(output['delta'].cpu().numpy())
            
        mu = np.concatenate(mu, axis = 0)
        pi = np.concatenate(pi, axis = 0)
        theta = torch.exp(self.model.decoder.log_theta).detach().cpu().numpy()
        mu_intrinsic = np.concatenate(mu_intrinsic, axis = 0)
        delta = np.concatenate(delta, axis = 0)
        return {
            'mu': mu,
            'pi': pi,
            'theta': theta,
            'mu_intrinsic': mu_intrinsic,
            'delta': delta
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