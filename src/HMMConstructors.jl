export GaussianHMM,
    SwitchingGaussianRegression, SwitchingBernoulliRegression, SwitchingPoissonRegression, SwitchingAutoRegression

"""
    GaussianHMM(; K::Int, output_dim::Int, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))

Create a Hidden Markov Model with Gaussian Emissions

# Arguments
- `K::Int`: The number of hidden states
- `output_dim::Int`: The dimensionality of the observation
- `A::Matrix{<:Real}=initialize_transition_matrix(K)`: The transition matrix of the HMM (defaults to random initialization)
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (defaults to random initialization)

# Returns
- `::HiddenMarkovModel`: Hidden Markov Model Object with Gaussian Emissions
```
"""
function GaussianHMM(;
    K::Int,
    output_dim::Int,
    A::AbstractMatrix{T},
    πₖ::AbstractVector{T},
    emissions::AbstractVector
) where {T<:Real}
    # Return constructed GaussianHMM
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end

"""
    SwitchingGaussianRegression(; 
        K::Int,
        input_dim::Int,
        output_dim::Int,
        include_intercept::Bool = true,
        β::Matrix{<:Real} = if include_intercept
            zeros(input_dim + 1, output_dim)
        else
            zeros(input_dim, output_dim)
        end,
        Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
        λ::Float64 = 0.0,
        A::Matrix{<:Real} = initialize_transition_matrix(K),
        πₖ::Vector{Float64} = initialize_state_distribution(K)
    )

Create a Switching Gaussian Regression Model

# Arguments
- `K::Int`: The number of hidden states.
- `input_dim::Int`: The dimensionality of the input features.
- `output_dim::Int`: The dimensionality of the output predictions.
- `include_intercept::Bool`: Whether to include an intercept in the regression model (default is `true`).
- `β::Matrix{<:Real}`: The regression coefficients (defaults to zeros based on `input_dim` and `output_dim`).
- `Σ::Matrix{<:Real}`: The covariance matrix of the Gaussian emissions (defaults to an identity matrix).
- `λ::Float64`: The regularization parameter for the regression (default is `0.0`).
- `A::Matrix{<:Real}`: The transition matrix of the Hidden Markov Model (defaults to random initialization).
- `πₖ::Vector{Float64}`: The initial state distribution of the Hidden Markov Model (defaults to random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching Gaussian Regression Model
"""
function SwitchingGaussianRegression(;
    K::Int,
    emissions::AbstractVector,
    A::AbstractMatrix,
    πₖ::AbstractVector,
)
    # Return the HiddenMarkovModel
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end

"""
    SwitchingAutoRegression(; K::Int, output_dim::Int, order::Int, include_intercept::Bool=true, β::Matrix{<:Real}=if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end, Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim), λ::Float64=0.0, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))

Create a Switching AutoRegression Model

# Arguments
- `K::Int`: The number of hidden states.
- `output_dim::Int`: The dimensionality of the output data.
- `order::Int`: The order of the autoregressive model.
- `include_intercept::Bool=true`: Whether to include an intercept in the regression model.
- `β::Matrix{<:Real}`: The autoregressive coefficients (defaults to zeros).
- `Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim)`: The covariance matrix for the autoregressive model (defaults to an identity matrix).
- `λ::Float64=0.0`: Regularization parameter for the regression (defaults to zero).
- `A::Matrix{<:Real}=initialize_transition_matrix(K)`: The transition matrix of the HMM (Defaults to a random initialization). 
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (Defaults to a random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching AutoRegression Model
"""
function SwitchingAutoRegression(;
    K::Int,
    A::AbstractMatrix,
    πₖ::AbstractVector,
    emissions::AbstractVector
)
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end

"""
    SwitchingBernoulliRegression(; K::Int, input_dim::Int, include_intercept::Bool=true, β::Vector{<:Real}=if include_intercept zeros(input_dim + 1) else zeros(input_dim) end, λ::Float64=0.0, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))

Create a Switching Bernoulli Regression Model

# Arguments
- `K::Int`: The number of hidden states.
- `input_dim::Int`: The dimensionality of the input data.
- `include_intercept::Bool=true`: Whether to include an intercept in the regression model (defaults to true).
- `β::Vector{<:Real}`: The regression coefficients (defaults to zeros). 
- `λ::Float64=0.0`: Regularization parameter for the regression (defaults to zero).
- `A::Matrix{<:Real}=initialize_transition_matrix(K)`: The transition matrix of the HMM (defaults to random initialization).
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (defaults to random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching Bernoulli Regression Model
"""
function SwitchingBernoulliRegression(;
    K::Int,
    A::AbstractMatrix,
    πₖ::AbstractVector,
    emissions::AbstractVector,
)
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end


function SwitchingPoissonRegression(;
    K::Int,
    A::AbstractMatrix,
    πₖ::AbstractVector,
    emissions::AbstractVector,
)
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end
