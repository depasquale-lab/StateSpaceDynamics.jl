export SLDS

"""
    SLDS{T,S,O,TM,ISV}

A Switching Linear Dynamical System (SLDS). A hierarchical time-series model of the form:

```math
z_t | z_{t-1} ~ Categorical(A_{z_{t-1}, :})
x_t | x_{t-1}, z_t ~ N(A^{(z_t)} x_{t-1} + b^{(z_t)}, Q^{(z_t)})
y_t | x_t, z_t ~ N(C^{(z_t)} x_t + d^{(z_t)}, R^{(z_t)})
```

# Fields
- `A::TM`: Transition matrix for the discrete states (K x K)
- `πₖ::ISV`: Initial state distribution for the discrete states (K-dimensional vector)
- `LDSs::Vector{LinearDynamicalSystem{T,S,O}}`: Vector of K Linear Dynamical Systems, one for each discrete state
"""
@kwdef mutable struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
} <: AbstractHMM
    A::TM
    πₖ::ISV
    LDSs::Vector{LinearDynamicalSystem{T,S,O}}
end

"""
    initialize_forward_backward(model::AbstractHMM, num_obs::Int)

Initialize the forward backward storage struct.
"""
function initialize_forward_backward(model::SLDS, num_obs::Int, ::Type{T}) where {T<:Real}
    num_states = size(model.A, 1)

    return ForwardBackward(
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_states),
    )
end

"""
    rand(rng::AbstractRNG, slds::SLDS{T,S,O}; tsteps::Int, ntrials::Int=1) where {T<:Real, S<:AbstractStateModel, O<:AbstractObservationModel}

Sample from a Switching Linear Dynamical System (SLDS). Returns a tuple `(z, x, y)` where:
- `z` is a matrix of discrete states of size `(tsteps, ntrials)`
- `x` is a 3D array of continuous latent states of size `(latent_dim, tsteps, ntrials)`
- `y` is a 3D array of observations of size `(obs_dim, tsteps, ntrials)`
"""
function Random.rand(
    rng::AbstractRNG, slds::SLDS{T,S,O}; tsteps::Int, ntrials::Int=1
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)  # Number of discrete states
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    # Pre-allocate outputs
    z = Array{Int,2}(undef, tsteps, ntrials)        # Discrete states
    x = Array{T,3}(undef, latent_dim, tsteps, ntrials)  # Continuous states  
    y = Array{T,3}(undef, obs_dim, tsteps, ntrials)     # Observations

    # Pre-extract parameters for all LDS models (avoid repeated extraction)
    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]

    # Sample each trial
    for trial in 1:ntrials
        _sample_slds_trial!(
            rng,
            view(z, :, trial),
            view(x,:,:,trial),
            view(y,:,:,trial),
            slds.A,
            slds.πₖ,
            state_params,
            obs_params,
            slds.LDSs[1].obs_model,
        )  # Use first for type dispatch
    end

    return z, x, y
end

# Core SLDS trial sampling logic
function _sample_slds_trial!(
    rng, z_trial, x_trial, y_trial, A, πₖ, state_params, obs_params, obs_model_type
)
    tsteps = length(z_trial)
    K = size(A, 1)

    # Sample discrete state sequence using forward sampling
    z_trial[1] = rand(rng, Categorical(πₖ))
    for t in 2:tsteps
        z_trial[t] = rand(rng, Categorical(A[z_trial[t - 1], :]))
    end

    # Sample continuous states and observations given discrete sequence
    return _sample_continuous_given_discrete!(
        rng, x_trial, y_trial, z_trial, state_params, obs_params, obs_model_type
    )
end

# Sample continuous dynamics given discrete state sequence
function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::GaussianObservationModel,
)
    tsteps = length(z_trial)

    # Initial state from the selected LDS
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand(
        rng, MvNormal(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d, obs_params[k1].R)
    )

    # Subsequent states - switch dynamics based on discrete state
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t - 1], z_trial[t]

        # Continuous state follows previous discrete state's dynamics
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_prev].A * x_trial[:, t - 1] + state_params[k_prev].b,
                state_params[k_prev].Q,
            ),
        )

        # Observation follows current discrete state's model
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d,
                obs_params[k_curr].R,
            ),
        )
    end
end

function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::PoissonObservationModel,
)
    tsteps = length(z_trial)

    # Initial state
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand.(
        rng, Poisson.(exp.(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d))
    )

    # Subsequent states
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t - 1], z_trial[t]

        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_prev].A * x_trial[:, t - 1] + state_params[k_prev].b,
                state_params[k_prev].Q,
            ),
        )

        y_trial[:, t] = rand.(
            rng, Poisson.(exp.(obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d))
        )
    end
end

# Convenience method without explicit RNG
Random.rand(slds::SLDS; kwargs...) = rand(Random.default_rng(), slds; kwargs...)

"""
    loglikelihood(slds::SLDS, x, y, w)
    
Compute weighted complete-data log-likelihood for SLDS.
Returns vector of per-timestep log-likelihoods.
"""
function loglikelihood(
    slds::SLDS{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},  # (K, tsteps)
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K, tsteps = size(w)
    ll_vec = zeros(T, tsteps)

    for k in 1:K
        # Get per-timestep log-likelihoods from LDS k
        ll_k = loglikelihood(x, slds.LDSs[k], y)  # Vector{T}

        # Weight by discrete state posterior
        ll_vec .+= w[k, :] .* ll_k
    end

    return ll_vec
end

function Gradient(
    slds::SLDS, y::AbstractMatrix{T}, x::AbstractMatrix{T}, w::AbstractMatrix{T}
) where {T<:Real}
    latent_dim, tsteps = size(x)
    grad = zeros(T, latent_dim, tsteps)
    K = length(slds.LDSs)

    for k in 1:K
        lds_k = slds.LDSs[k]

        # Compute unweighted gradient for LDS k
        grad_k = Gradient(lds_k, y, x)

        # Weight each timestep by discrete state posterior
        for t in 1:tsteps
            grad[:, t] .+= w[k, t] .* grad_k[:, t]
        end
    end

    return grad
end

function Hessian(
    slds::SLDS, y::AbstractMatrix{T}, x::AbstractMatrix{T}, w::AbstractMatrix{T}
) where {T<:Real}
    K = length(slds.LDSs)
    latent_dim, tsteps = size(x)

    # Initialize block-tridiagonal structure
    H_diag_total = [zeros(T, latent_dim, latent_dim) for _ in 1:tsteps]
    H_sub_total = [zeros(T, latent_dim, latent_dim) for _ in 1:(tsteps - 1)]
    H_super_total = [zeros(T, latent_dim, latent_dim) for _ in 1:(tsteps - 1)]

    for k in 1:K
        lds_k = slds.LDSs[k]

        # Compute unweighted Hessian blocks for LDS k
        _, H_diag_k, H_super_k, H_sub_k = Hessian(lds_k, y, x)

        # Weight diagonal blocks by discrete state posterior at time t
        for t in 1:tsteps
            H_diag_total[t] .+= w[k, t] .* H_diag_k[t]
        end

        # Weight off-diagonal blocks by discrete state posterior at time t+1
        # (where the dynamics land)
        for t in 1:(tsteps - 1)
            H_sub_total[t] .+= w[k, t + 1] .* H_sub_k[t]
            H_super_total[t] .+= w[k, t + 1] .* H_super_k[t]
        end
    end

    H = block_tridgm(H_diag_total, H_super_total, H_sub_total)
    return H, H_diag_total, H_super_total, H_sub_total
end

function smooth!(
    slds::SLDS, fs::FilterSmooth{T}, y::AbstractMatrix{T}, w::AbstractMatrix{T}
) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    tsteps = size(y, 2)

    # Initial guess from previous iteration or zeros
    X₀ = Vector{T}(vec(fs.E_z))

    # Define weighted negative log-likelihood
    function nll(vec_x::AbstractVector{T})
        x = reshape(vec_x, latent_dim, tsteps)
        return -sum(loglikelihood(slds, x, y, w))
    end

    # Weighted gradient
    function g!(grad::Vector{T}, vec_x::Vector{T})
        x = reshape(vec_x, latent_dim, tsteps)
        grad_mat = Gradient(slds, y, x, w)
        grad .= vec(-grad_mat)
        return nothing
    end

    # Weighted Hessian
    function h!(h::SparseMatrixCSC{T}, vec_x::Vector{T})
        x = reshape(vec_x, latent_dim, tsteps)
        H, _, _, _ = Hessian(slds, y, x, w)
        mul!(h, -1.0, H)
        return nothing
    end

    # Setup optimization
    initial_f = nll(X₀)
    initial_g = similar(X₀)
    g!(initial_g, X₀)
    initial_h = spzeros(T, length(X₀), length(X₀))
    h!(initial_h, X₀)

    td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, initial_g, initial_h)
    opts = Optim.Options(; g_abstol=1e-8, x_abstol=1e-8, f_abstol=1e-8, iterations=100)

    # Optimize
    res = optimize(td, X₀, Newton(; linesearch=LineSearches.BackTracking()), opts)

    # Store result
    fs.x_smooth .= reshape(res.minimizer, latent_dim, tsteps)

    # Compute covariances
    H, main, super, sub = Hessian(slds, y, fs.x_smooth, w)

    if latent_dim > 10
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse(
            -sub, -main, -super
        )
    else
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse_static(
            -sub, -main, -super, Val(latent_dim)
        )
    end

    fs.p_smooth .= p_smooth_result
    fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result

    # Symmetrize
    @views for t in 1:tsteps
        fs.p_smooth[:, :, t] .= 0.5 .* (fs.p_smooth[:, :, t] .+ fs.p_smooth[:, :, t]')
    end

    return fs
end

# Public API wrapper
function smooth(slds::SLDS, y::AbstractMatrix{T}, w::AbstractMatrix{T}) where {T<:Real}
    fs = initialize_FilterSmooth(slds.LDSs[1], size(y, 2))
    smooth!(slds, fs, y, w)
    return fs.x_smooth, fs.p_smooth
end

"""
    sample_posterior(rng::AbstractRNG, fs::FilterSmooth{T}) where {T<:Real}

Sample a trajectory from the posterior over continuous states and compute its entropy.

Returns:
- x_sample: matrix of size (latent_dim, tsteps) representing one sample from 
  q(x) = ∏ₜ N(x_t | x_smooth_t, p_smooth_t)
- entropy: H[q(x)] = ∑ₜ H[N(x_t | x_smooth_t, p_smooth_t)]
"""
function sample_posterior(rng::AbstractRNG, fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)
    x_sample = similar(fs.x_smooth)
    entropy = zero(T)
    min_jitter = T(1e-8)

    for t in 1:tsteps
        μ = fs.x_smooth[:, t]
        Σ = Symmetric(fs.p_smooth[:, :, t])

        # Try Cholesky decomposition with increasing jitter if needed
        chol = nothing
        jitter = zero(T)
        max_attempts = 5

        for attempt in 1:max_attempts
            try
                chol = cholesky(Σ + jitter * I)
                break
            catch e
                if attempt == max_attempts
                    # Last resort: use larger jitter
                    jitter = min_jitter * T(10)^(attempt-1)
                    @warn "Covariance matrix not positive definite at t=$t, adding jitter=$jitter"
                    chol = cholesky(Σ + jitter * I)
                else
                    # Increase jitter and try again
                    jitter = min_jitter * T(10)^(attempt-1)
                end
            end
        end

        # Sample using the Cholesky factor
        Σ_chol = chol.L
        x_sample[:, t] = μ + Σ_chol * randn(rng, T, latent_dim)

        # Accumulate entropy using log determinant from Cholesky
        # log|Σ| = 2 * sum(log(diag(L))) where Σ = L*L'
        logdet_Σ = 2 * sum(log, diag(Σ_chol))
        entropy += 0.5 * (latent_dim * (1 + log(2π)) + logdet_Σ)
    end

    return x_sample, entropy
end

# Convenience method
function sample_posterior(fs::FilterSmooth{T}) where {T<:Real}
    return sample_posterior(Random.GLOBAL_RNG, fs)
end

"""
    sample_posterior(rng::AbstractRNG, tfs::TrialFilterSmooth{T}, nsamples::Int=1) where {T<:Real}

Sample trajectories from the posterior for multiple trials and compute entropies.

Returns:
- samples: array of size (latent_dim, tsteps, ntrials, nsamples)
- entropies: matrix of size (ntrials, nsamples) containing entropy for each sample
"""
function sample_posterior(
    rng::AbstractRNG, tfs::TrialFilterSmooth{T}, nsamples::Int=1
) where {T<:Real}
    ntrials = length(tfs.FilterSmooths)
    latent_dim, tsteps = size(tfs[1].x_smooth)

    samples = Array{T,4}(undef, latent_dim, tsteps, ntrials, nsamples)
    entropies = Matrix{T}(undef, ntrials, nsamples)

    for trial in 1:ntrials
        for s in 1:nsamples
            samples[:, :, trial, s], entropies[trial, s] = sample_posterior(rng, tfs[trial])
        end
    end

    return samples, entropies
end

function sample_posterior(tfs::TrialFilterSmooth{T}, nsamples::Int=1) where {T<:Real}
    return sample_posterior(Random.GLOBAL_RNG, tfs, nsamples)
end

"""
    estep!(slds::SLDS, tfs::TrialFilterSmooth, fbs::Vector{ForwardBackward}, y::AbstractArray, x_samples::AbstractArray)

E-step for SLDS using a single sample from the continuous posterior.
- Uses sampled continuous states to compute emission likelihoods
- Runs forward-backward to get discrete state posteriors  
- Smooths continuous states given discrete posteriors
- Computes sufficient statistics
"""
function estep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fbs::AbstractVector{<:ForwardBackward},
    y::AbstractArray{T,3},
    x_samples::AbstractArray{T,4},  # (latent_dim, tsteps, ntrials, nsamples=1)
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    ntrials = size(y, 3)
    K = length(slds.LDSs)
    tsteps = size(y, 2)

    total_elbo = zero(T)

    for trial in 1:ntrials
        y_trial = view(y,:,:,trial)
        x_sample = view(x_samples,:,:,trial,1)  # Use first (and only) sample
        fb = fbs[trial]  # Use trial-specific ForwardBackward

        for k in 1:K
            fb.loglikelihoods[k, :] .= loglikelihood(x_sample, slds.LDSs[k], y_trial)
        end

        # Run forward-backward for discrete states
        forward!(slds, fb)
        backward!(slds, fb)
        calculate_γ!(slds, fb)
        calculate_ξ!(slds, fb)

        # Get discrete state posteriors (normalize from log space)
        w = exp.(fb.γ)  # (K, tsteps)

        # Smooth continuous states given discrete posteriors
        smooth!(slds, tfs[trial], y_trial, w)

        # Compute sufficient statistics for M-step
        sufficient_statistics!(tfs[trial])

        # Compute trial contribution to ELBO
        # ELBO = E_q(z)q(x)[log p(y,x,z)] - H[q(x)] - H[q(z)]
        trial_elbo = zero(T)

        # E_q(z)q(x)[log p(y,x,z)] = sum_k q(z_t=k) * log p(y_t,x_t|z_t=k)
        # Use updated x_smooth for computing complete-data likelihood
        x_smooth_trial = tfs[trial].x_smooth

        for k in 1:K
            # Complete-data log-likelihood for LDS k: log p(y,x|z=k)
            # This already includes state dynamics + observations
            ll_k = loglikelihood(x_smooth_trial, slds.LDSs[k], y_trial)  # Vector of length tsteps

            # Weight by discrete state posterior at each time
            for t in 1:tsteps
                trial_elbo += w[k, t] * ll_k[t]
            end
        end

        # Discrete state prior: log p(z)
        # Initial state
        trial_elbo += sum(w[k, 1] * log(slds.πₖ[k] + 1e-12) for k in 1:K)

        # Transitions
        for i in 1:K, j in 1:K
            trial_elbo += exp(fb.ξ[i, j]) * log(slds.A[i, j] + 1e-12)
        end

        # Subtract entropies
        trial_elbo -= tfs[trial].entropy  # H[q(x)]

        # H[q(z)] = -sum_t sum_k q(z_t=k) log q(z_t=k)
        discrete_entropy =
            -sum(w[k, t] * log(w[k, t] + 1e-12) for k in 1:K, t in 1:tsteps if w[k, t] > 0)
        trial_elbo -= discrete_entropy

        total_elbo += trial_elbo
    end

    return total_elbo
end

"""
    mstep!(slds::SLDS, tfs::TrialFilterSmooth, fbs::Vector{ForwardBackward}, y::AbstractArray)

M-step for SLDS.
- Updates discrete HMM parameters (A, Z₀) using aggregated statistics across trials
- Updates each LDS using weighted sufficient statistics
"""
function mstep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fbs::AbstractVector{<:ForwardBackward{T}},
    y::AbstractArray{T,3},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    ntrials = size(y, 3)
    tsteps = size(y, 2)

    # Update HMM parameters using aggregated statistics from all trials
    update_initial_state_distribution!(slds, fbs)
    update_transition_matrix!(slds, fbs)

    # Update each LDS using weighted data across all trials
    for k in 1:K
        # Collect weights for state k from all trials as a vector of vectors
        weights = Vector{Vector{T}}(undef, ntrials)
        for trial in 1:ntrials
            weights[trial] = exp.(fbs[trial].γ[k, :])
        end

        # Update LDS k with weighted sufficient statistics
        mstep!(slds.LDSs[k], tfs, y, weights)
    end

    return nothing
end

"""
    fit!(slds::SLDS, y::AbstractArray; max_iter=25, progress=true)

Fit SLDS using variational Laplace EM algorithm with stochastic ELBO estimates.
Runs for exactly max_iter iterations (no early stopping due to stochastic estimates).
"""
function fit!(
    slds::SLDS{T,S,O}, y::AbstractArray{T,3}; max_iter::Int=50, progress::Bool=true
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    ntrials = size(y, 3)
    tsteps = size(y, 2)
    K = length(slds.LDSs)

    # Initialize structures
    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps, ntrials)
    fbs = [initialize_forward_backward(slds, tsteps, T) for _ in 1:ntrials]

    # Initialize progress bar
    prog = if progress
        Progress(max_iter; desc="Fitting SLDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    # Storage for ELBO values
    elbos = Vector{T}(undef, max_iter)

    # Initialize with uniform weights and smooth once
    w_uniform = ones(T, K, tsteps) ./ K
    for trial in 1:ntrials
        smooth!(slds, tfs[trial], y[:, :, trial], w_uniform)
    end

    # Main EM loop - runs for exactly max_iter iterations
    for iter in 1:max_iter
        # Sample from current continuous posteriors
        x_samples, entropies = sample_posterior(Random.default_rng(), tfs, 1)

        # E-step: infer discrete states and update continuous states
        elbo = estep!(slds, tfs, fbs, y, x_samples)
        elbos[iter] = elbo

        # M-step: update parameters
        mstep!(slds, tfs, fbs, y)

        # Update progress
        if progress && prog !== nothing
            next!(prog; showvalues=[(:iteration, iter), (:ELBO, elbo)])
        end
    end

    # Finish progress bar
    if progress && prog !== nothing
        finish!(prog)
    end

    return elbos
end
