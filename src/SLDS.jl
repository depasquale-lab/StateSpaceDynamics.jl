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
- `Z₀::ISV`: Initial state distribution for the discrete states (K-dimensional vector)
- `LDSs::Vector{LinearDynamicalSystem{T,S,O}}`: Vector of K Linear Dynamical Systems, one for each discrete state
"""
@kwdef struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
} <: AbstractHMM
    A::TM
    Z₀::ISV
    LDSs::Vector{LinearDynamicalSystem{T,S,O}}
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
            slds.Z₀,
            state_params,
            obs_params,
            slds.LDSs[1].obs_model,
        )  # Use first for type dispatch
    end

    return z, x, y
end

# Core SLDS trial sampling logic
function _sample_slds_trial!(
    rng, z_trial, x_trial, y_trial, A, Z₀, state_params, obs_params, obs_model_type
)
    tsteps = length(z_trial)
    K = size(A, 1)

    # Sample discrete state sequence using forward sampling
    z_trial[1] = rand(rng, Categorical(Z₀))
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
    fs.entropy = gaussian_entropy(Symmetric(H))

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

function mstep!(slds::SLDS, 
                tfs::TrialFilterSmooth, 
                fb::ForwardBackward, 
                y::AbstractMatrix)

    # update hmm parameters
    update_initial_state_distribution!(slds, fb)
    update_transmision_matrix!(slds, fb)
    

    # update LDS paramerers
    for (k, lds) in enumerate(slds.LDSs)
        weights = w[k, :]
        mstep!(lds, tfs, y, weights)
    end

    return nothing
end
