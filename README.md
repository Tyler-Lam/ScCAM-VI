# Single Cell Cross-Attention Decoder Variational Inference

An idea I had to use cross-attention mechanisms in a VAE decoder to learn spatial relations between cell gene expressions, and model spatial associations of specific genes between specific cell phenotypes using ablation frameworks.

# Mathematical overview

Gene counts are modeled using a zero-inflated negative-binomial distribution with parameters $(\mu_{i,j,k}, \theta_i, \pi_i)$, where

$$
\mu_{i,j,k} = \text{Mean of gene } i \text{ in cell } j \text{ with phenotype label }k
$$
$$
\theta_i = \text{Dispersion parameter for gene } i
$$
$$
\pi_i = \text{Zero inflation parameter for gene } i
$$

### Spatially invariant VAE
##### Encoder
The model uses a residual architecture to first estimate the baseline gene expression for a cell of type $k$ given the input gene expression for cell $j$. The encoder takes the input of cell $j$ and creates an embedding with celltype aware gaussian prior given by

$$
p(\mathbf{z}\mid k) = \mathcal{N}(\mathbf{m}_k, 1)
$$

with a prior learned mean $\mathbf{m}_k$ for each cell phenotype label. The posterior distribution for the latent distribution is given by

$$
q(\mathbf{z}\mid \mathbf{x})=\mathcal{N}(\mathbf{\mu}_\text{emb}, \mathbf{\sigma}^2) \qquad \mathbf{z} = \mathbf{\mu}_\text{emb} + \mathbf{\epsilon} * \mathbf{\sigma}\qquad \epsilon\sim\mathcal{N}(0,1)
$$
