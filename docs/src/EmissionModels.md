# What are Emission Models?

```@meta
CollapsedDocStrings = true
```

**Emission models** describe how observations are generated from latent states in a state space model. These models define the conditional distribution of the observed data given the hidden state or input features. In `StateSpaceDynamics.jl`, a flexible suite of emission models is supported, including both simple parametric distributions and regression-based models.

At a high level, emission models encode:

- **The distribution of observations** (e.g., Gaussian, Poisson, Bernoulli)
- **How observations relate to inputs or latent states**, either directly or via regression

```@docs
EmissionModel
RegressionEmission
```

## Gaussian Emission Model

The **GaussianEmission** is a basic model where the observations are drawn from a multivariate normal distribution with a fixed mean and covariance.

```math
y_t \sim \mathcal{N}(\mu, \Sigma)
```

This emission model is often used when the observed data is real-valued and homoscedastic.

```@docs
GaussianEmission
loglikelihood(::GaussianEmission, ::AbstractMatrix)
Random.rand(::GaussianEmission)
fit!(::GaussianEmission, ::AbstractMatrix)
```

## Regression-Based Emission Models

Regression-based emissions allow the output to depend on an input matrix ``\Phi``. The regression relationship is defined by a coefficient matrix ``\beta``, optionally with an intercept and regularization.

### Gaussian Regression Emission

In the **GaussianRegressionEmission**, the outputs are real-valued and modeled via linear regression with additive Gaussian noise.

```math
y_t \sim \mathcal{N}(\Phi_t \beta, \Sigma)
```

```@docs
GaussianRegressionEmission
Random.rand(::GaussianRegressionEmission, ::Union{Matrix, Vector})
loglikelihood(::GaussianRegressionEmission, ::AbstractMatrix, ::AbstractMatrix)
fit!(::GaussianRegressionEmission, ::AbstractMatrix, ::AbstractMatrix)
```

### Bernoulli Regression Emission

The **BernoulliRegressionEmission** is appropriate for binary data. The probability of success is modeled via a logistic function.

```math
p(y_t = 1 \mid \Phi_t) = \sigma(\Phi_t \beta)
```

Where ``\sigma(z) = 1 / (1 + e^{-z})`` is the logistic function.

```@docs
BernoulliRegressionEmission
Random.rand(::BernoulliRegressionEmission, ::Union{Matrix, Vector})
loglikelihood(::BernoulliRegressionEmission, ::AbstractMatrix, ::AbstractMatrix)
```

### Poisson Regression Emission

The **PoissonRegressionEmission** is ideal for count data, such as spike counts in neuroscience. It models the intensity of the Poisson distribution as an exponential function of the linear predictors.

```math
y_t \sim \text{Poisson}(\lambda_t), \quad \lambda_t = \exp(\Phi_t \beta)
```

```@docs
PoissonRegressionEmission
Random.rand(::PoissonRegressionEmission, ::Union{Matrix, Vector})
loglikelihood(::PoissonRegressionEmission, ::AbstractMatrix, ::AbstractMatrix)
```

## Autoregressive Emission Models

The **AutoRegressionEmission** models the observation at time `t` as depending on previous observations (i.e., an autoregressive structure), using a wrapped `GaussianRegressionEmission`.

```math
y_t \sim \mathcal{N}(\sum_{i=1}^p A_i y_{t-i}, \Sigma)
```

Where `p` is the autoregressive order and `A_i` are regression weights.

This model is useful when modeling temporal dependencies in the emission process, independent of latent dynamics.

```@docs
AutoRegressionEmission
construct_AR_feature_matrix
Random.rand(::AutoRegressionEmission, ::AbstractMatrix)
loglikelihood(::AutoRegressionEmission, ::AbstractMatrix, ::AbstractMatrix)
```

## Fitting Regression Emission Models

All regression-based emissions can be fitted using maximum likelihood with optional weights and L2 regularization. Internally, `StateSpaceDynamics.jl` formulates this as an optimization problem, solved using gradient-based methods (e.g., LBFGS).

```@docs
fit!(::RegressionEmission, ::AbstractMatrix, ::AbstractMatrix)
create_optimization
RegressionOptimization
```
