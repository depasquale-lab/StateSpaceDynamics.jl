# Basic Models
```@meta
CollapsedDocStrings = true
```

## Gaussian
```@docs
Gaussian
sample(model::Gaussian; n::Int=1)
loglikelihood(model::Gaussian, Y::Matrix{<:Real})
fit!(model::Gaussian, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1))) 
```