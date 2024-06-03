# Mixture Models

A [mixture model](http://en.wikipedia.org/wiki/Mixture_model) is a probabilistic distribution that combines a set of *components* to represent the overall distribution. Generally, the probability density/mass function is given by a convex combination of the pdf/pmf of individual components, as

```math
f_{mix}(x; \Theta, \pi) = \sum_{k=1}^K \pi_k f(x; \theta_k)
```

A *mixture model* is characterized by a set of component parameters ``\Theta=\{\theta_1, \ldots, \theta_K\}`` and a prior distribution ``\pi`` over these components.

## Structs
```@docs
GaussianMixtureModel
PoissonMixtureModel
```