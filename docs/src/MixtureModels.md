# Mixture Models
```@meta
CollapsedDocStrings = true
```

A [mixture model](http://en.wikipedia.org/wiki/Mixture_model) is a probability distribution which, given a finite ``k > 0``, samples from ``k`` different distributions ``\{f_i(x) | i \in \{1,...,k\}\}`` randomly, where the probability of sampling from ``f_i(x)`` is ``\pi_i``. Generally, a mixture model is written in the form of:

```math
f_{mix}(x; \Theta, \pi) = \sum_{k=1}^K \pi_k f_k(x)
```

Where ``f_i(x)`` is called the ith *component* and ``\pi_i`` is called the ith *mixing coeffiecent*.



## Gaussian Mixture Model
```@docs
GaussianMixtureModel
GaussianMixtureModel(k::Int, data_dim::Int)
fit!(gmm::GaussianMixtureModel, data::Matrix{<:Real}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=false)
log_likelihood(gmm::GaussianMixtureModel, data::Matrix{<:Real})
sample(gmm::GaussianMixtureModel, n::Int)
```

## Poisson Mixture Model
```@docs
PoissonMixtureModel
PoissonMixtureModel(k::Int)
fit!(pmm::PoissonMixtureModel, data::Matrix{Int64}; maxiter::Int=50, tol::Float64=1e-3,initialize_kmeans::Bool=false)
log_likelihood(pmm::PoissonMixtureModel, data::Matrix{Int64})
sample(pmm::PoissonMixtureModel, n::Int)
```