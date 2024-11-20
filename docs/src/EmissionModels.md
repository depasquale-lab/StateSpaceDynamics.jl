# Emission Models

```@meta
CurrentModule = StateSpaceDynamics
```

The `StateSpaceDynamics.jl` package provides several emission models for state space modeling. These models define how observations are generated from latent states.

## Basic Emission Models

### Gaussian Emissions

A basic Gaussian emission model with mean and covariance parameters. This model is suitable for continuous data that follows a normal distribution.

```@docs
GaussianEmission
```

The Gaussian emission model can be constructed with:
```julia
GaussianEmission(; 
    output_dim::Int, 
    μ::Vector{<:Real}=zeros(output_dim),
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim)
)
```

#### Methods
```@docs
sample(model::GaussianEmission, n::Int=1)
loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})
fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64})
```

## Regression-based Emissions

Regression-based emissions model the relationship between input features and observations through different probability distributions.

### Gaussian Regression

A Gaussian regression emission model that relates input features to output observations through a linear transformation with Gaussian noise.

```@docs
GaussianRegressionEmission
```

Example constructor usage:
```julia
regression = GaussianRegressionEmission(
    input_dim=3,
    output_dim=2,
    include_intercept=true,
    λ=0.1  # L2 regularization
)
```

#### Methods
```@docs
sample(model::GaussianRegressionEmission, Φ::Matrix{<:Real})
loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
fit!(model::GaussianRegressionEmission, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64})
```

### Bernoulli Regression

A Bernoulli regression emission model for binary outcomes, using a logistic link function.

```@docs
BernoulliRegressionEmission
```

Example usage for binary classification:
```julia
binary_model = BernoulliRegressionEmission(
    input_dim=4,
    output_dim=1,
    include_intercept=true
)

# Fit to binary data
X_train = randn(100, 4)
y_train = rand(Bool, 100, 1)
fit!(binary_model, X_train, float.(y_train))
```

#### Methods
```@docs
sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real})
loglikelihood(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
fit!(model::BernoulliRegressionEmission, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64})
```

### Poisson Regression

A Poisson regression emission model for count data, using a log link function.

```@docs
PoissonRegressionEmission
```

Example usage for count data:
```julia
count_model = PoissonRegressionEmission(
    input_dim=3,
    output_dim=1,
    include_intercept=true
)

# Fit to count data
X_train = randn(100, 3)
y_train = rand(Poisson(2), 100, 1)
fit!(count_model, X_train, float.(y_train))
```

#### Methods
```@docs
sample(model::PoissonRegressionEmission, Φ::Matrix{<:Real})
loglikelihood(model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
fit!(model::PoissonRegressionEmission, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64})
```

## Autoregressive Emissions

### AutoRegression

An autoregressive emission model that depends on previous observations, useful for time series data.

```@docs
AutoRegressionEmission
```

Example usage:
```julia
ar_model = AutoRegressionEmission(
    output_dim=2,
    order=3,
    include_intercept=true
)

# Generate samples using previous observations
Y_prev = randn(3, 2)  # 3 previous observations for order=3
new_sample = sample(ar_model, Y_prev)
```

#### Methods
```@docs
sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real})
loglikelihood(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64})
```

## Composite Models

### Composite Emissions

A wrapper for combining multiple emission models into a single model.

```@docs
CompositeModelEmission
```

Example of combining models:
```julia
# Create individual models
gaussian = GaussianEmission(output_dim=2)
regression = GaussianRegressionEmission(input_dim=3, output_dim=2)

# Combine into composite model
composite = CompositeModelEmission([gaussian, regression])
```

#### Methods
```@docs
sample(model::CompositeModelEmission, input_data::Vector{})
loglikelihood(model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{})
fit!(model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{}, w::Vector{Float64})
```