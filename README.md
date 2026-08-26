# Single Cell Cross-Attention Model Variational Inference

An idea I had to use cross-attention mechanisms in a VAE decoder to learn spatial relations between cell gene expressions, and model spatial associations of specific genes between specific cell phenotypes using ablation frameworks.

# Mathematical overview

Gene counts are modeled using a zero-inflated negative-binomial distribution with parameters $(\mu_{g,j,k}, \theta_g, \pi_g)$, where

$$
\mu_{g,j,k} = \text{Mean of gene } g \text{ in cell } j \text{ with phenotype label }k
$$
$$
\theta_g = \text{Dispersion parameter for gene } g
$$
$$
\pi_g = \text{Zero inflation parameter for gene } g
$$

### Spatially invariant VAE
The model uses a residual architecture to first estimate the baseline gene expression for a cell of type $k$ given the input gene expression for cell $j$. The encoder is a standard MLP that takes the input of cell $j$ and creates an embedding with celltype aware gaussian prior given by

$$
p(\mathbf{z}\_i\mid k) = \mathcal{N}(\mathbf{m}\_{i,k}, 1)
$$

with a prior learned mean $\mathbf{m}\_{i,k}$, where $z\in[1,n\_\text{latent}]$. The intrinsic embedding is resampled from the posterior distribution with a learned mean $\mu_\text{int,i}$ and variance $\sigma_i$. A decoder then takes this intrinsic embedding and outputs the estimated mean of the ZINB distribution $\hat{\mu}_{g,\mathrm{int}}$

### Residual Spatial Attention Block
The aggregated neighboring cells are passed through the spatially invariant encoder to obtain the latent representation. For N neighboring cells, we obtain the neighbor latent embeddings and distances to the central cell

$$
\lbrace z_{\text{int},n},\ d_{n}\\rbrace , \quad n\in[1, N]
$$

In the cross attention block, the query is given by the central cell while the keys and values are given by the neighboring cells. The distance is encoded as an additive bias to the attention logits using a radial basis function with learnable weights. This allows the model to freely determine the influence of cells at given distances. The final context from the attention block $c_i$ goes through a separate MLP decoder to obtain $\delta$, which is treated as a residual correction to the intrinsic gene estimation. The spatially aware mean estimation is then given by

$$
\hat{\mu}\_{g}=\hat{\mu}\_{g,\mathrm{int}}\cdot\exp{(\alpha\cdot\delta_g)}
$$

The parameter $\alpha$ is a ramping coefficient that is initially 0 to first train the intrinsic VAE to reconstruct gene expression with no spatial context. In this formulation, the spatial attention $\delta$ can be thought of as a LFC correction to the intrinsic gene estimation. The total loss is given by

$$
\mathcal{L}=\mathcal{L}\_\text{recon} + \gamma\cdot\mathcal{L}\_{\text{recon},\text{intrinsic}} + \beta\_{kl}\cdot\mathcal{L}_{kl} + \lambda\_{m}\cdot\mathcal{L}\_\text{m} + \lambda\_\delta\cdot\mathcal{L}\_\delta
$$

The first term $\mathcal{L}\_\text{recon}$ is the negative log-likelihood of the ZINB using the spatially aware mean gene estimations. The second term $\mathcal{L}\_{\text{recon},\text{intrinsic}}$ is the negative log-likelihood of the ZINB using the intrinsic mean gene estimation, with an annealing parameter $\gamma$ which ramps down from 1 to encourage the model to learn from the spatial context. The third term $\mathcal{L}_{kl}$ is the standard KL divergence on the celltype aware latent representation with annealing parameter $\beta\_{kl}$. The fourth term $\mathcal{L}\_\text{m}=\Sigma\_k \mid m\_k\mid$ is L1 regularization on the celltype prior means $m\_k$ with coefficient $\lambda\_{m}$. The fifth term $\mathcal{L}\_\delta$ is L1 regularization on the spatial log-fold change to the intrinsic mean gene expression with coefficient given by $\lambda\_\delta$. When a cell type is strongly spatially autocorrelated, the model can learn to use the spatial information as a more stable estimate of intrinsic gene expression as opposed to estimating the gene from the intrinsic representation. This regularization is designed to discourage this.

## Ablation Studies
One key goal of this model is to determine the association of neighboring cell types with a central cell's gene expression program. By itself, the residual model allows us to directly see the deviation from a "baseline" resulting from spatial context using neighboring cells. However, this architecture lends itself well to ablation studies in order to determine the effect of specific cell types on the expression program. By treating each cell type as a "player" and the attention model $\delta$ as a value function, we can calculate the SHAP scores for neighboring cell types and determine the marginalized contribution of each cell type to the LFC from baseline gene expression. This is currently done using permutation based on cell type labels, where cell types are permuted to form a random coalition, then ablated sequentially to estimate their marginal contribution. The three ways to ablate a cell type that are currently implement are "zeroing" where the embedding is replaced with a zero tensor, "masking" where the embedding is masked from the attention block (assigned -inf bias), or "mean" where the embedding is replaced by the average cell embedding.

## Train/Validation Splitting and Data Leakage
Cell neighbors are aggregated using a fixed radius, then binned into hexagonal grids. Both the radius and hexagonal grid size are parameters given by user input. Bins are then randomly assigned train, test, or validation based on given data splitting arguments. To prevent data leakage, a central cell must have all neighbors in the same category. Otherwise a central cell in the training dataset may be used as a neighboring cell in validation. Central cells with neighbors in different categories can still be used as neighboring cells.

## Development
This model is currently very underdeveloped. The base architecture produces reasonable embeddings for clustering 10x Xenium breast cancer data, and clustering on the attention model context produces spatially coherent clusters, but the architecture and ablation methods will likely need tuning. Several of the loss annealing parameters are ad-hoc fixes to suboptimal architecture, and I'm sure there's a better way for the loss to reflect the residual architecture than the current implementation. The main issue is that the attention block and spatial decoder are not trained with ablation in mind. If all neighbors are ablated, the context of the attention output is fixed to the zero tensor. The spatial decoder is not encouraged to output a delta of 0 given a zero tensor as input, and so the baseline level from ablation is inaccurate. Additionally, masking other cells will increase the influence from other cells due to how the softmax normalization scales the attention logits. If we instead zero the ablated cells, this can be out of distribution (especially with the learned cell type specific priors). Using the mean cell type seems to be the best option, but this is heavily influenced by the overall distribution of the training and validation data. 

Additionally, where to actually put layer_norm for the attention context is an open question. The current architecture has the option of projecting the embeddings before passing through the attention block and using layer_norm on the attention context. However, several attention models use layer_norm on the inputs to the attention block and use the raw output. Additionally, I've considered adding variational framework to match the intrinsic embeddings, but I have no idea if that's proper or would work. Lots of stuff to tune here.