import sys
sys.path.append('/common/lamt2/attention/src/')
import pandas as pd
from neighbors import *
from training import *
from ablation import *
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.special import factorial
import seaborn as sns
from pathlib import Path
import os
import time
from tqdm import tqdm
import seaborn as sns

dirs = ['run_07_10_26_all_neighbors']
for d in dirs:
    
    dst_dir = f'/common/lamt2/attention/outs/{d}'
    if not os.path.isdir(dst_dir):
        Path(dst_dir).mkdir(parents = True, exist_ok=True)

    adata = sc.read_h5ad('/common/lamt2/attention/data/anndata/adata_merged_qc.h5ad')

    neighbors = SpatialNeighbors(
        adata,
        unique_core_key='sample',
        neighbor_method = 'radius',
        radius = 200,
    )

    neighbors.build_neighbors()

    split_idx = neighbors.split_data(bin_length = 700, verbose = True)

    trainer = SpatialAutoencoderTrainer(
        graph = neighbors,
        batch_key = 'sample',
        celltype_key = 'celltype',
        layer = 'counts',
        hidden_dims = [128],
        latent_dim = 8,
        attn_dim = 64,
        project_inputs=True,
        topk = -1,
        batch_dim = 2,
        rbf_n_basis = 10,
        rbf_spacing = 'linear',
        num_heads = 4,
        max_epochs = 500,
        beta_ramp_start = 25,
        beta_ramp_end = 75,
        beta_kl_max = 10,
        alpha_ramp_start = 100,
        alpha_ramp_end = 150,
        alpha_max = 1.0,
        lambda_delta = 1e-4,
        lambda_mu_prior = 5e-3,
        gamma_ramp_start = 150,
        gamma_ramp_end = 200,
        gamma = 0.5,
        early_stop_patience = 10,
        early_stop_offset = 20,
        num_workers = 0,
    )

    trainer.setup()

    #trainer.train()
    trainer.model.load_state_dict(torch.load(f'{dst_dir}/model.pt', weights_only = True))

    ablation = SpatialAblation(trainer, batch_size = 1024)

    scores = ablation.run_ablation_mc(category = 'celltype', n_iter = 1000, method = 'mean')
    for c in scores:
        np.save(f'{dst_dir}/scores__mean___{c}.npy', scores[c])
        
    print('\n\n ---- Done ---- \n')