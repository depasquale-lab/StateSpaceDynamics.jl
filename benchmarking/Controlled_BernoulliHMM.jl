using Distributions

import HiddenMarkovModels as HMMs

export ControlledBernoulliHMM, format_glmhmm_data

# Helper function: Converts a vector to a matrix with a given shape
function vec_to_matrix(vec::Vector, shape::Tuple{Int, Int})
    reshape(vec, shape)
end

mutable struct ControlledBernoulliHMM{T} <: HMMs.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dist_coeffs::Vector{Vector{T}}  # One vector for each state
end

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

# Regression opt storage for HMM.jl benchmark
struct RegressionOptimizationTest{}
    X::Matrix{}
    y::Matrix{}
    w::Vector{}
    β_shape::Tuple{Int, Int}
end

# Bernoulli Regression Objective Function
function objective(opt::RegressionOptimizationTest, β_vec)
    β_mat = vec_to_matrix(β_vec, opt.β_shape) # Reshape vector to matrix
    p = logistic.(opt.X * β_mat)             # Predicted probabilities
    # Calculate negative log-likelihood
    val = -sum(opt.w .* (opt.y .* log.(p) .+ (1 .- opt.y) .* log.(1 .- p)))
    return val
end

function objective_gradient!(
    G::Vector{Float64},
    opt::RegressionOptimizationTest,
    β_vec::Vector{Float64})
    β_mat = vec_to_matrix(β_vec, opt.β_shape)


    p = logistic.(opt.X * β_mat)

    grad_mat = -(opt.X' * (opt.w .* (opt.y .- p)))
    return G .= vec(grad_mat)
end


function fit_bern(X::Matrix{<:Real}, y::Matrix{<:Real}, β_shape::Tuple{Int, Int}, β_init::Vector{<:Real}, w::Vector{Float64}=ones(size(y,1)))
    # Create the RegressionOptimization struct
    opt_problem = RegressionOptimizationTest(X, y, w, β_shape)
    f(β) = objective(opt_problem, β)
    g!(G, β) = objective_gradient!(G, opt_problem, β)

    # Set optimization options
    opts = Optim.Options(;
        x_abstol=1e-8,
        x_reltol=1e-8,
        f_abstol=1e-8,
        f_reltol=1e-8,
        g_abstol=1e-8,
    )


    # Run optimization
    result = optimize(f, g!, β_init, LBFGS(), opts)
    # Retrieve the optimized parameters
    β_opt_vec = result.minimizer
    β_opt_mat = vec_to_matrix(β_opt_vec, β_shape)

    return β_opt_mat

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
    updated_betas = Vector{Vector{Float64}}(undef, N)  # To store new coefficients for each state
    @threads for i in 1:N
        # Weight observations by state responsibility (γ)
        state_weights = γ[i, :]

        # Get right shapes for fit_bern
        β_shape = size(hmm.dist_coeffs[i])[1]
        β_init = vec(hmm.dist_coeffs[i])
        control_matrix = hcat(control_seq...)
        obs_matrix = reshape(obs_seq, :, 1)
        control_matrix = permutedims(control_matrix)
        result = fit_bern(control_matrix, obs_matrix,(β_shape, 1), β_init, state_weights)

        # Store updated coefficients
        updated_betas[i] = vec(result)  # Store the updated coefficients as a vector
    end

    # Update the HMM's coefficients with the newly fitted values
    hmm.dist_coeffs = updated_betas

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

function HiddenMarkovModels.baum_welch_has_converged(
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
