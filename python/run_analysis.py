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

dst_dir = '/common/lamt2/attention/outs/run_07_10_26_all_neighbors'
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

trainer.train()
trainer.save(dst_dir)
fig = trainer.plot_history(show = False)
plt.savefig(f'{dst_dir}/loss.png', bbox_inches = 'tight')
plt.close()

centers, widths, weights = trainer.model.attention.distance_encoding.get_distance_curve()
plt.scatter(centers, weights)
plt.xlabel('Distance [um]')
plt.ylabel('Attention bias')
plt.savefig(f'{dst_dir}/rbf_distances.png')
plt.close()

embs = trainer.get_embedding()
np.save(f'{dst_dir}/X_intrinsic.npy', embs['z_intrinsic'])
np.save(f'{dst_dir}/X_spatial.npy', embs['z_spatial'])

fig, ax = plt.subplots(2, 1, figsize = (6.4, 4.8 * 2), constrained_layout = True)
sns.violinplot(embs['z_intrinsic'], ax = ax[0], inner = 'quart')
ax[0].set_title("Z_intrinsic")
sns.violinplot(embs['z_spatial'], ax = ax[1], inner = 'quart')
ax[1].set_title("Z_spatial")
plt.savefig(f'{dst_dir}/violins.png')
plt.close()

predictions = trainer.predict()
deltas = predictions['delta'].flatten()
dmin = np.quantile(deltas, .01)
dmax = np.quantile(deltas, .99)
plt.hist(deltas, bins = np.linspace(dmin, dmax, 50))
plt.yscale("log")
plt.xlabel("Spatial LFC")
plt.ylabel("Counts")
plt.savefig(f'{dst_dir}/lfcs.png')
plt.close()

trainer.adata.obsm['X_vae'] = embs['z_intrinsic']
print("Building neighborhood graph ... ", end = '')
t0 = time.perf_counter()
sc.pp.neighbors(trainer.adata, use_rep = 'X_vae', n_neighbors = 20)
print(f'done: {(time.perf_counter() - t0) / 60:.2f} s')
t0 = time.perf_counter()
print("Calculating UMAP ... ", end = '')
umap_args = {
    'min_dist': 0.1,
    'spread': 1,
    'init_pos': 'spectral',
}
sc.tl.umap(trainer.adata, **umap_args)
print(f"done: {(time.perf_counter() - t0) / 60:.2f} s")
for r in tqdm([0.3, 0.5, 1.0], desc = 'Leiden clustering'):
    sc.tl.leiden(
        trainer.adata,
        resolution = r,
        random_state= 42,
        flavor = 'igraph',
        n_iterations = 2,
        directed=False,
        key_added = f'leiden_{str(r).replace(".", "p")}'
    )

t0 = time.perf_counter()
trainer.adata.obsm['X_niche'] = embs['z_spatial']
print("Clustering niches ... ", end = '')
sc.pp.neighbors(trainer.adata, use_rep = 'X_niche', key_added = 'niche_neighbors', n_neighbors = 20)
sc.tl.leiden(
    trainer.adata,
    resolution = 0.5,
    random_state = 42,
    flavor = 'igraph',
    n_iterations = 2,
    directed = False,
    neighbors_key = 'niche_neighbors',
    key_added = 'niche'
)
print(f'done: {(time.perf_counter() - t0) / 60:.2f} s')
sc.pl.umap(
    trainer.adata,
    color = ['celltype', 'cluster', 'leiden_0p3', 'leiden_0p5', 'leiden_1p0', 'niche', 'split_category'],
    wspace = 0.2,
    show = False
)
plt.savefig(f'{dst_dir}/umap.png', bbox_inches = 'tight', dpi = 200)
plt.close()

sc.pl.scatter(
    trainer.adata,
    x = 'x_centroid',
    y = 'y_centroid',
    color = ['celltype', 'cluster', 'leiden_0p3', 'leiden_0p5', 'leiden_1p0', 'niche', 'split_category'],
    show = False
)
plt.savefig(f'{dst_dir}/spatial.png', bbox_inches = 'tight', dpi = 200)
plt.close()

trainer.save(dst_dir)
