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
        max_neighbors: int = 50,
        
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
        max_beta_kl: float = 5,
        beta_ramp_start: int = 10,
        beta_ramp_end: int = 30,
        alpha_ramp_start: int = 50,
        alpha_ramp_end: int = 100,
        
        # Training kwargs
        max_epochs: int = 1000,
        grad_clip_norm: float = 1.0,
        
        # Other kwargs:
        device: str = 'auto',
        dst_dir: Optional[str] = None,
        random_state: int = 42
    ):
        self.graph = graph
        
        self.random_state = random_state
        self.batch_key = batch_key
        self.layer = layer
        self.max_neighbors = max_neighbors

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
        self.max_beta_kl = max_beta_kl
        self.beta_ramp_start = beta_ramp_start
        self.beta_ramp_end = beta_ramp_end
        self.alpha_ramp_start = alpha_ramp_start
        self.alpha_ramp_end = alpha_ramp_end
        
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
            layer = self.layer,
            batch_key = f'{self.batch_key}_int' if self.batch_key else None,
            random_state = self.random_state            
        )

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

        train_graph, train_idx = self.graph.slice_by_index(self.graph.split_idx['train_idx'])
        self.train_loader = DataLoader(
            SpatialDataset.from_graph(
                train_graph,
                cell_indices = train_idx, 
                batch_key = f'{self.batch_key}_int' if self.batch_key else None,
                layer = self.layer,
                max_neighbors = self.max_neighbors,
                train = True,
                random_state = self.random_state,
            ),
            sampler = make_stratified_sampler(self.graph.split_idx['train_idx']),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = (self.device.type == 'cuda'),
        )
        
        test_graph, test_idx = self.graph.slice_by_index(self.graph.split_idx['test_idx'])
        self.test_loader = DataLoader(
            SpatialDataset.from_graph(
                test_graph,
                cell_indices = test_idx,
                batch_key = f'{self.batch_key}_int' if self.batch_key else None,
                layer = self.layer,
                train = False,
                random_state = self.random_state
            ),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = (self.device.type == 'cuda'),
            drop_last = False,
            shuffle = False
        )
        
        val_graph, val_idx = self.graph.slice_by_index(self.graph.split_idx['val_idx'])
        self.val_loader = DataLoader(
            SpatialDataset.from_graph(
                val_graph,
                cell_indices = val_idx,
                batch_key = f'{self.batch_key}_int' if self.batch_key else None,
                layer = self.layer,
                random_state = self.random_state
            ),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = (self.device.type == 'cuda'),
            drop_last = False,
            shuffle = False
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = (self.device.type == 'cuda'),
            shuffle = False
        )
        
    def _setup_model(self):
        if self.graph is None:
            self._setup_spatial_graph()
        
        self.model = SpatialAutoEncoder(
            n_genes = self.graph.adata.shape[1],
            latent_dim = self.latent_dim,
            hidden_dims = self.hidden_dims,
            num_heads = self.num_heads,
            dropout = self.dropout,
            n_batches = self.n_batches,
            batch_dim = self.batch_dim,
            d_min  = self.graph.adata.obsm[self.graph.distance_key].data.min(),
            d_max = self.graph.adata.obsm[self.graph.distance_key].data.max(),
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
        
    def train(self, random_state: int = 42, verbose: bool = True):
        
        set_random_seed(random_state, self.device)
        history = []
        for epoch in (pbar := tqdm(range(self.max_epochs), desc = 'Training model', disable = not verbose)):
            
            # Training loop
            n_train = 0
            train_loss = 0.0    
            train_recon = 0.0
            train_kl = 0.0   
            beta_kl = get_anneal_ramp_param(epoch, self.beta_ramp_start, self.beta_ramp_end, self.max_beta_kl, method = 'cosine')
            alpha = get_anneal_ramp_param(epoch, self.alpha_ramp_start, self.alpha_ramp_end, 1, method = 'cosine')
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                
                # Update progress bar description in 20% batch increments
                if (i % max(1, len(self.train_loader) // 5)) == 0 or (i == len(self.train_loader) - 1):
                    pbar.set_description(f'Running training loop: batch = {i+1}/{len(self.train_loader)} (Early stop count = {self.early_stopping.counter})')
                
                cell_X = batch['cell_X'].to(self.device)
                neighbor_X = batch['neighbor_X'].to(self.device)
                neighbor_mask = batch['neighbor_mask'].to(self.device)
                distances = batch['distances'].to(self.device)
                log_library_size = batch['log_library_size'].to(self.device)
                batch_label = batch['batch_label'].to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    central_X = cell_X,
                    neighbor_X = neighbor_X,
                    neighbor_mask = neighbor_mask,
                    distances = distances,
                    log_library_size = log_library_size,
                    batch_label = batch_label,
                    alpha = alpha
                )
                
                loss, recon_loss, kl_loss = self.loss(
                    outputs['mu_z'],
                    outputs['log_var'],
                    outputs['mu_x'],
                    outputs['theta'],
                    outputs['pi'],
                    cell_X,
                    beta_kl,
                )
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm = self.grad_clip_norm
                )
                self.optimizer.step()
                
                n = len(cell_X)
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
            pbar.set_description(f'Running validation loop (Early stop count = {self.early_stopping.counter})')

            self.model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                # Update progress bar description in 20% batch increments
                    cell_X = batch['cell_X'].to(self.device)
                    neighbor_X = batch['neighbor_X'].to(self.device)
                    neighbor_mask = batch['neighbor_mask'].to(self.device)
                    distances = batch['distances'].to(self.device)
                    log_library_size = batch['log_library_size'].to(self.device)
                    batch_label = batch['batch_label'].to(self.device)
                    outputs = self.model(
                        central_X = cell_X,
                        neighbor_X = neighbor_X,
                        neighbor_mask = neighbor_mask,
                        distances = distances,
                        log_library_size = log_library_size,
                        batch_label = batch_label,
                        alpha = alpha
                    )
                    
                    loss, recon_loss, kl_loss = self.loss(
                        outputs['mu_z'],
                        outputs['log_var'],
                        outputs['mu_x'],
                        outputs['theta'],
                        outputs['pi'],
                        cell_X,
                        beta_kl
                    )

                    n = len(cell_X)
                    n_val += n
                    val_loss += loss.item() * n
                    val_recon += recon_loss * n
                    val_kl += kl_loss * n

                val_loss /= n_val
                val_recon /= n_val
                val_kl /= n_val
            
            if epoch > self.beta_ramp_start:
                self.scheduler.step(val_loss)
            if epoch > max(self.alpha_ramp_end, self.beta_ramp_end) + self.early_stop_offset:
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
                'alpha': alpha
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
            cell_X = batch['cell_X'].to(self.device)
            neighbor_X = batch['neighbor_X'].to(self.device)
            neighbor_mask = batch['neighbor_mask'].to(self.device)
            distances = batch['distances'].to(self.device)
            log_library_size = batch['log_library_size'].to(self.device)
            batch_label = batch['batch_label'].to(self.device)
            outputs = self.model(
                central_X = cell_X,
                neighbor_X = neighbor_X,
                neighbor_mask = neighbor_mask,
                distances = distances,
                log_library_size = log_library_size,
                batch_label = batch_label,
            )
            
            loss, recon_loss, kl_loss = self.loss(
                outputs['mu_z'],
                outputs['log_var'],
                outputs['mu_x'],
                outputs['theta'],
                outputs['pi'],
                cell_X,
                self.max_beta_kl
            )
            
            n = len(cell_X)
            n_test += n
            test_loss += n * loss.item()
            test_recon += n * recon_loss
            test_kl += n * kl_loss
        test_loss /= n_test
        test_recon /= n_test
        test_kl /= n_test
        return test_loss, test_recon, test_kl
    
    @torch.no_grad()
    def get_embedding(self):
        self.model.eval()
        embeddings = {'pre_attention': [], 'post_attention': []}
        cell_indices = []
        
        for batch in self.dataloader:
            cell_X = batch['cell_X'].to(self.device)
            neighbor_X = batch['neighbor_X'].to(self.device)
            neighbor_mask = batch['neighbor_mask'].to(self.device)
            distances = batch['distances'].to(self.device)
            mu, log_var, pre_z, post_z = self.model.encode(
                central_X = cell_X,
                neighbor_X = neighbor_X,
                neighbor_mask = neighbor_mask,
                distances = distances
            )
            
            embeddings['pre_attention'].append(pre_z.cpu().numpy())
            embeddings['post_attention'].append(post_z.cpu().numpy())
            cell_indices.append(batch['cell_idx'])
            
        self.embedding = {
            'pre_attention': np.concatenate(embeddings['pre_attention'], axis = 0),
            'post_attention': np.concatenate(embeddings['post_attention'], axis = 0),
            'cell_idx': np.concatenate(cell_indices, axis = 0)
        }
        
        return self.embedding
    
    @torch.no_grad()
    def predict(self):
        self.model.eval()
        x_hat = []
        pi = []
        t1 = time.perf_counter()
        for batch in self.dataloader:
            cell_X = batch['cell_X'].to(self.device)
            neighbor_X = batch['neighbor_X'].to(self.device)
            neighbor_mask = batch['neighbor_mask'].to(self.device)
            distances = batch['distances'].to(self.device)
            log_library_size = batch['log_library_size'].to(self.device)
            batch_label = batch['batch_label'].to(self.device)
            result = self.model(
                central_X = cell_X,
                neighbor_X = neighbor_X,
                neighbor_mask = neighbor_mask,
                distances = distances,
                log_library_size = log_library_size,
                batch_label = batch_label
            )
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
        self.graph.adata.write_h5ad(f'{dst_dir}/adata.h5ad')