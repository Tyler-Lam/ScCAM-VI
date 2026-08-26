# Single Cell Cross-Attention Decoder Variational Inference

An idea I had to use cross-attention mechanisms in a VAE decoder to learn spatial relations between cell gene expressions, and model spatial associations of specific genes between specific cell phenotypes using ablation frameworks.

### Math

Gene counts are modeled using a zero-inflated negative-binomial distribution with parameters $(\mu_{i,j,k}, \theta_i, \pi_i)$, where

$$
\mu_{i,j} = \text{Mean of gene } i \text{ in cell } j \text{ with phenotype label }k
$$
$$
\theta_i = \text{Dispersion parameter for gene } i
$$
$$
\pi_i = \text{Zero inflation parameter for gene } i
$$
