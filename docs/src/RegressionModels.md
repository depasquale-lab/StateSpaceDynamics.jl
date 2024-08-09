# Regression Models
```@meta
CollapsedDocStrings = true
```

The following Regression Models are common Generalized Linear Models (GLM) with their corresponding canonical links.



## Gaussian Regression
```@docs
GaussianRegression
sample(model::GaussianRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
loglikelihood(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
fit!(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
```

## Bernoulli Regression
```@docs
BernoulliRegression
sample(model::BernoulliRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
loglikelihood(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
fit!(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
```

## Poisson Regression
```@docs
PoissonRegression
sample(model::PoissonRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
loglikelihood(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
fit!(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
```

## Auto Regression
```@docs
AutoRegression
sample(model::AutoRegression, Y_prev::Matrix{<:Real}; n::Int=1)
loglikelihood(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
fit!(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
```