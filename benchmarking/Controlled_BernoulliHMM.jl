using Distributions

import HiddenMarkovModels as HMMs

export format_glmhmm_data

using Distributions
import HiddenMarkovModels as HMMs

logistic(x::Real) = 1 / (1 + exp(-x))
logistic(x::AbstractArray) = 1 ./ (1 .+ exp.(-x))

export ControlledBernoulliHMM, format_glmhmm_data

# Define the model struct
mutable struct ControlledBernoulliHMM{T} <: HMMs.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dist_coeffs::Vector{Vector{T}}  # One vector for each state
end

# HMM function definitions
function HMMs.initialization(hmm::ControlledBernoulliHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::ControlledBernoulliHMM, control::AbstractVector)
    return hmm.trans
end

# Modified to use Bernoulli with probabilities from logistic regression
function HMMs.obs_distributions(hmm::ControlledBernoulliHMM, control::AbstractVector)
    return [
        Bernoulli(logistic(dot(hmm.dist_coeffs[i], control))) for i in 1:length(hmm)
    ]
end

function objective(dist_coeffs::AbstractVector, obs_seq::AbstractVector, control::AbstractVector, weights::AbstractVector)
    X = hcat(control...)'  # T × D matrix
    logits = X * dist_coeffs
    p = logistic.(logits)
    val = -sum(weights .* (obs_seq .* log.(p) .+ (1 .- obs_seq) .* log.(1 .- p)))
    return val
end

function objective_gradient!(
    G::AbstractVector,
    dist_coeffs::AbstractVector,
    obs_seq::AbstractVector,
    control::AbstractVector,
    weights::AbstractVector,
)
    X = hcat(control...)'  # T × D
    logits = X * dist_coeffs
    p = logistic.(logits)
    residuals = weights .* (p .- obs_seq)  # shape (T,)
    grad = X' * residuals  # shape (D,)
    return G .= grad

end

function StatsAPI.fit!(
    hmm::ControlledBernoulliHMM{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends,
) where {T}
    (; γ, ξ) = fb_storage
    N = length(hmm)

    # Update initial probabilities and transition matrix
    hmm.init .= 0
    hmm.trans .= 0
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= γ[:, t1]
        hmm.trans .+= sum(ξ[t1:t2])
    end
    hmm.init ./= sum(hmm.init)
    for row in eachrow(hmm.trans)
        row ./= sum(row)
    end

    """
    Update Bernoulli Regression Coefficients for Each State
    """

    for k = 1:length(hmm)
        # val = objective(hmm.dist_coeffs[k], obs_seq, control_seq, γ[k, :])
        # println(val)
        result = optimize(
        β -> objective(β, obs_seq, control_seq, γ[k, :]),
        (G, β) -> objective_gradient!(G, β, obs_seq, control_seq, γ[k, :]),
        hmm.dist_coeffs[k],
        BFGS(); inplace = true
        )

        hmm.dist_coeffs[k] .= Optim.minimizer(result)
    end
end

function HMMs.baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev

        if progress < atol
            return true
        end
    end
    return false
end


function format_glmhmm_data(X::Vector{Matrix{Float64}}, Y::Vector{Matrix{Float64}})
    obs_seq = Vector{Float64}()
    control_seq = Vector{Vector{Float64}}()
    seq_ends = Int[]

    for (x, y) in zip(X, Y)
        T = size(x, 2)  # Number of timepoints
        append!(obs_seq, vec(y))  # Flatten 1×T matrix into T-vector
        append!(control_seq, [x[:, t] for t in 1:T])  # Push T vectors of length 3
        push!(seq_ends, length(obs_seq))  # Track cumulative timepoints
    end

    return obs_seq, control_seq, seq_ends
end
