# Hidden Markov Models

A Hidden Markov Model (HMM) is a time series model with unobservable latent variables that follow a Markov process. Condititional distributions (when conditioned on the latents) are often chosen to be common distributions (e.g. Gaussian).

## Gaussian HMM
```@docs
GaussianHMM{GaussianEmission}
GaussianHMM(A::Matrix{Float64}, B::Vector{GaussianEmission}, πₖ::Vector{Float64}, K::Int, D::Int)
GaussianHMM(data::Matrix{Float64}, k_states::Int=2)
baumWelch!(hmm::AbstractHMM, data::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
```


## Switching Gaussian Regression
```@docs
SwitchingGaussianRegression
SwitchingGaussianRegression(; num_features::Int, num_targets::Int, A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
sample(model::SwitchingGaussianRegression, X::Matrix{Float64})
```
```@docs; canonical=false
fit!(model::hmmglm, X::Matrix{Float64}, y::Union{Vector{T}, BitVector, Matrix{Float64}}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true) where T<: Real
```

## Switching Bernoulli Regression
```@docs
SwitchingBernoulliRegression
SwitchingBernoulliRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
```
```@docs; canonical=false
fit!(model::hmmglm, X::Matrix{Float64}, y::Union{Vector{T}, BitVector, Matrix{Float64}}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true) where T<: Real
```

## Switching Poisson Regression
```@docs
SwitchingPoissonRegression
SwitchingPoissonRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
```
```@docs; canonical=false
fit!(model::hmmglm, X::Matrix{Float64}, y::Union{Vector{T}, BitVector, Matrix{Float64}}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true) where T<: Real
```

## Viterbi Algorithm
```@docs
viterbi(hmm::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
```

