# Regression Models

The following Regression Models are common Generalized Linear Models (GLM) with their corresponding canonical links.



## Gaussian Regression
```@docs
GaussianRegression
loglikelihood(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64})
fit!(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64}, w::Vector{Float64}=ones(size(y, 1)))
sample(model::GaussianRegression, X::Matrix{Float64})
```

## Bernoulli Regression
```@docs
BernoulliRegression
loglikelihood(model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}, BitVector}, w::Vector{Float64}=ones(length(y)))
loglikelihood(model::BernoulliRegression, X::Vector{Float64}, y::Union{Float64, Bool, Int64}, w::Float64=1.0)
fit!(model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}, BitVector}, w::Vector{Float64}=ones(length(y)))
```

## Poisson Regression
```@docs
PoissonRegression
loglikelihood(model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
loglikelihood(model::PoissonRegression, X::Vector{Float64}, y::Union{Float64, Int64}, w::Float64=1.0)
fit!(model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
```