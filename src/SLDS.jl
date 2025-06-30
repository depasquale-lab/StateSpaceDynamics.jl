export SwitchingLinearDynamicalSystem, fit!, initialize_slds, variational_expectation!

"""
Switching Linear Dynamical System

Struct to Encode a Hidden Markov model that switches among K distinct LinearDyanmicalSystems

# Fields 
- `A::V`: Transition matrix for mode switching. 
- `B::VL`: Vector of Linear Dynamical System models. 
- `πₖ::V`: Initial state distribution.
- `K::Int`: Number of modes. 
"""
mutable struct SwitchingLinearDynamicalSystem{T<:Real,M<:AbstractMatrix{T}, V<:AbstractVector{T}, VL<:AbstractVector{<:LinearDynamicalSystem}} <: AbstractHMM
    A::M 
    B::VL 
    πₖ::V  
    K::Int 
end

"""
    Random.rand(rng, slds, T)

Generate synthetic data with switching LDS models

#Arguments 
- `rng:AbstractRNG`: Random number generator
- `slds::SwitchingLinearDynamicalSystem`: The switching LDS model 
- `T::Int`: Number of time steps to sample

# Returns
- `Tuple{Array,Array, Array}`: Latent states (x), observations (y), and mode sequences (z). 
"""
function Random.rand(rng::AbstractRNG, slds::SwitchingLinearDynamicalSystem, T::Int)
    state_dim = slds.B[1].latent_dim
    obs_dim = slds.B[1].obs_dim
    K = slds.K

    x = zeros(state_dim, T)  # Latent states
    y = zeros(obs_dim, T)    # Observations
    z = zeros(Int, T)        # Mode sequence

    # Sample initial mode
    z[1] = rand(rng, Categorical(slds.πₖ / sum(slds.πₖ)))
    x[:, 1] = rand(rng, MvNormal(zeros(state_dim), slds.B[z[1]].state_model.Q))
    y[:, 1] = rand(rng, MvNormal(slds.B[z[1]].obs_model.C * x[:, 1], slds.B[z[1]].obs_model.R))

    @views for t in 2:T
        # Sample mode based on transition probabilities
        z[t] = rand(Categorical(slds.A[z[t-1], :] ./ sum(slds.A[z[t-1], :])))
        # Update latent state and observation
        x[:, t] = rand(
          MvNormal(slds.B[z[t]].state_model.A * x[:, t-1], slds.B[z[t]].state_model.Q)
        )
        y[:, t] = rand(
          MvNormal(slds.B[z[t]].obs_model.C * x[:, t], slds.B[z[t]].obs_model.R)
        )
    end

    return x, y, z
end

function Random.rand(slds::SwitchingLinearDynamicalSystem, T::Int)
    return rand(Random.default_rng(), slds, T)
end

"""
    initialize_slds(;K::Int=2, d::Int=2, p::Int=10, self_bias::Float64=5.0, seed::Int=42)

Initialize a Switching Linear Dynamical System with random parameters.
"""
function initialize_slds(;K::Int=2, d::Int=2, p::Int=10, self_bias::Float64=5.0, seed::Int=42)
    Random.seed!(seed)
    
    # Transition matrix using Dirichlet with self-bias
    A = zeros(K, K)
    for i in 1:K
        # Create concentration parameters with higher value for self-transition
        alpha = ones(K)
        alpha[i] = self_bias  # Bias toward self-transition
        
        # Sample from Dirichlet distribution
        A[i, :] = rand(Dirichlet(alpha))
    end
    
    # Initial state probabilities
    πₖ = rand(Dirichlet(ones(K)))
    
    # State parameters
    Q = Matrix(0.001 * I(d))
    x0 = zeros(d)
    P0 = Matrix(0.001 * I(d))
    
    # Define observation parameters separately for each state
    B = Vector{LinearDynamicalSystem}(undef, K)
    
    # Generate state matrices with different dynamics for each state
    for k in 1:K
        # Create rotation angles for each 2D subspace in the state space
        F = Matrix{Float64}(I, d, d)
        
        # Parameter to make each state model unique
        angle_factor = 2π * (k-1) / K
        
        # Add rotation components in 2D subspaces
        for i in 1:2:d-1
            if i+1 <= d  # Ensure we have a pair
                # Create a 2D rotation with different angles for each state
                theta = 0.1 + angle_factor + (i-1)*0.2
                rotation = 0.95 * [cos(theta) -sin(theta); sin(theta) cos(theta)]
                F[i:i+1, i:i+1] = rotation
            end
        end
        
        # If d is odd, add a scaling factor to the last dimension
        if d % 2 == 1
            F[d, d] = 0.95
        end
        
        # Create state model with the designed dynamics
        state_model = GaussianStateModel(F, Q, x0, P0)
        
        # Observation matrix - random for each state
        C = randn(p, d)
        R = Matrix(0.001 * I(p))
        obs_model = GaussianObservationModel(C, R)
        
        # Create linear dynamical system for this state
        B[k] = LinearDynamicalSystem(state_model, obs_model, d, p, fill(true, 6))
    end
    
    return SwitchingLinearDynamicalSystem(A, B, πₖ, K)
end

"""
    initialize_FilterSmooth_slds(slds::SwitchingLinearDynamicalSystem{T}, T_step::Int) where {T<:Real}

Initialize FilterSmooth objects for each regime in the SLDS.
"""
function initialize_FilterSmooth_slds(slds::SwitchingLinearDynamicalSystem{T}, T_step::Int) where {T<:Real}
    K = slds.K
    FS = Vector{FilterSmooth{T}}(undef, K)
    
    for k in 1:K
        num_states = slds.B[k].latent_dim
        FS[k] = FilterSmooth(
            zeros(T, num_states, T_step),           # x_smooth
            zeros(T, num_states, num_states, T_step), # p_smooth
            zeros(T, num_states, T_step, 1),         # E_z
            zeros(T, num_states, num_states, T_step, 1), # E_zz
            zeros(T, num_states, num_states, T_step, 1)  # E_zz_prev
        )
    end
    
    return FS
end

"""
    smooth_with_weights!(fs::FilterSmooth{T}, lds::LinearDynamicalSystem{T,S,O}, 
                        y::AbstractMatrix{T}, weights::AbstractVector{T}) where {T<:Real,S,O}

Perform smoothing with weights and update the FilterSmooth object in-place.
"""
function smooth_with_weights!(
    fs::FilterSmooth{T}, 
    lds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractMatrix{T}, 
    weights::AbstractVector{T}
) where {T<:Real,S,O}
    # Perform smoothing with weights
    x_smooth, p_smooth, inverse_offdiag, total_entropy = smooth(lds, y, weights)
    
    # Update the FilterSmooth object
    fs.x_smooth .= x_smooth
    fs.p_smooth .= p_smooth
    
    # Compute sufficient statistics and store them in FilterSmooth
    fs.E_z[:, :, 1] .= x_smooth
    
    # Compute E_zz and E_zz_prev
    tsteps = size(x_smooth, 2)
    @views for t in 1:tsteps
        xt = x_smooth[:, t]
        pt = p_smooth[:, :, t]
        fs.E_zz[:, :, t, 1] .= pt .+ xt * xt'
        if t > 1
            xtm1 = x_smooth[:, t-1]
            pt1 = inverse_offdiag[:, :, t]
            fs.E_zz_prev[:, :, t, 1] .= pt1 .+ xt * xtm1'
        end
    end
    fs.E_zz_prev[:, :, 1, 1] .= 0
    
    return total_entropy
end

"""
    fit!(slds::SwitchingLinearDynamicalSystem, y::Matrix{T}; 
         max_iter::Int=1000, 
         tol::Real=1e-12, 
         ) where {T<:Real}

Fit a Switching Linear Dynamical System using the variational Expectation-Maximization (EM) algorithm with Kalman smoothing.

# Arguments
- `slds::SwitchingLinearDynamicalSystem`: The Switching Linear Dynamical System to be fitted.
- `y::Matrix{T}`: Observed data, size (obs_dim, T_steps).

# Keyword Arguments
- `max_iter::Int=1000`: Maximum number of EM iterations.
- `tol::Real=1e-12`: Convergence tolerance for log-likelihood change.

# Returns
- `mls::Vector{T}`: Vector of log-likelihood values for each iteration.
- `param_diff::Vector{T}`: Vector of parameter differences over each iteration. 
- `FB::ForwardBackward`: ForwardBackward struct 
- `FS::Vector{FilterSmooth}`: Vector of FilterSmooth structs
"""
function fit!(
    slds::AbstractHMM, y::AbstractMatrix{T}; max_iter::Int=1000, tol::Real=1e-3
) where {T<:Real}

    # Initialize log-likelihood
    prev_ml = -T(Inf)

    # Create a vector to store the log-likelihood values
    mls = Vector{T}()
    param_diff = Vector{T}()

    sizehint!(mls, max_iter)  # Pre-allocate for efficiency

    prog = Progress(
        max_iter; desc="Fitting SLDS via vEM...", barlen=50, showspeed=true
    )

    K = slds.K
    T_step = size(y, 2)
    
    # Initialize the FilterSmooth and ForwardBackward structs
    FB = initialize_forward_backward(slds, T_step, T)
    FS = initialize_FilterSmooth_slds(slds, T_step)

    # Initialize the parameters by running the Kalman Smoother for each model
    FB.γ = log.(ones(size(y)) * 0.5)

    # Run the Kalman Smoother for each model
    for k in 1:slds.K
        weights = exp.(FB.γ[k,:])
        smooth_with_weights!(FS[k], slds.B[k], y, weights)
    end

    # Run EM
    for i in 1:max_iter
        # E-step
        ml, _ = variational_expectation!(slds, y, FB, FS)       

        # M-step
        Δparams = mstep!(slds, FS, y, FB)
    
        # Update the log-likelihood vector and parameter difference
        push!(mls, ml)
        push!(param_diff, Δparams)

        # Update the progress bar
        next!(prog)

        # Check convergence
        if abs(ml - prev_ml) < tol
            finish!(prog)
            return mls, param_diff, FB, FS
        end

        prev_ml = ml
    end

    # Finish the progress bar if max_iter is reached
    finish!(prog)

    return mls, param_diff, FB, FS
end

"""
    variational_expectation!(model::SwitchingLinearDynamicalSystem, y, FB, FS) -> Float64

Compute the variational expectation (Evidence Lower Bound, ELBO) for a Switching Linear Dynamical System.
"""
function variational_expectation!(
    model::SwitchingLinearDynamicalSystem, 
    y::AbstractMatrix{T}, 
    FB::ForwardBackward, 
    FS::Vector{FilterSmooth{T}}
) where {T<:Real}
    # For now a hardcoded tolerance
    tol = 1e-6
    # Get starting point for iterative E-step
    γ = FB.γ
    hs = exp.(γ)
    ml_total = 0.
    # Initialize to something higher than the tolerance
    ml_diff = 1
    ml_prev = -Inf
    ml_storage = []
    
    while abs(ml_diff) > tol
        #1. compute qs from xs, which will live as log_likelihoods in FB
        variational_qs!([model.B[k].obs_model for k in 1:model.K], FB, y, FS)
        
        #2. compute hs from qs, which will live as γ in FB
        forward!(model, FB)
        backward!(model, FB)
        calculate_γ!(model, FB)
        calculate_ξ!(model, FB)   # needed for m_step update of transition matrix
        hs = exp.(FB.γ)
        ml_total = 0.0

        # ELBO from the discrete state model
        hmm_contribution = hmm_elbo(model, FB)
        ml_total += hmm_contribution
        
        for k in 1:model.K
            # Smooth with current weights and update FilterSmooth object
            weights = vec(hs[k,:])
            total_entropy = smooth_with_weights!(FS[k], model.B[k], y, weights)

            # Calculate the ELBO contribution for the current SSM
            # Convert single-trial format to multi-trial format for compatibility
            y_3d = reshape(y, (size(y)..., 1))
            elbo = calculate_elbo(
                model.B[k], 
                FS[k].E_z, 
                FS[k].E_zz, 
                FS[k].E_zz_prev, 
                reshape(FS[k].p_smooth, (size(FS[k].p_smooth)..., 1)), 
                y_3d, 
                total_entropy, 
                weights
            )
            ml_total += elbo
        end
       
        push!(ml_storage, ml_total)

        # Calculate difference between current and previous ml_total
        ml_diff = ml_total - ml_prev
        ml_prev = ml_total
    end
    
    return ml_total, ml_storage
end

"""
    hmm_elbo(model::AbstractHMM, FB::ForwardBackward; ϵ::Float64=1e-10)

Compute the evidence based lower bound (ELBO) from the discrete state model. 
"""
function hmm_elbo(model::AbstractHMM, FB::ForwardBackward; ϵ::Float64=1e-10)
    # Extract necessary data
    γ = FB.γ
    ξ = FB.ξ
    loglikelihoods = FB.loglikelihoods
    A = model.A   # Transition matrix
    πₖ = model.πₖ # Initial state distribution
    time_steps = size(loglikelihoods, 2)
    
    # Apply small epsilon to avoid log(0)
    safe_A = clamp.(A, ϵ, 1.0)
    safe_πₖ = clamp.(πₖ, ϵ, 1.0)
    
    # Initial state probabilities
    log_p_x_z = sum(exp.(γ[:, 1]) .* log.(safe_πₖ))
    
    # Transition term using ξ
    for t in 1:(time_steps - 1)
        log_p_x_z += sum(exp.(ξ[:, :, t]) .* log.(safe_A))
    end
    
    # Emission term
    log_p_x_z += sum(exp.(γ) .* loglikelihoods)
    
    # 2. Compute log q(z)
    # Here too we need to avoid log(0) for γ entries
    safe_γ = clamp.(γ, -log(1/ϵ), log(1/ϵ))  # Limit extremes in log space
    log_q_z = sum(exp.(γ) .* safe_γ)
    
    log_p_x_z - log_q_z
end

"""
    variational_qs!(obs_models, FB::ForwardBackward, y::AbstractMatrix{T}, FS::Vector{FilterSmooth{T}}) where {T<:Real}

Compute the variational distributions (`qs`) and update the log-likelihoods.
"""
function variational_qs!(
    obs_models::AbstractVector{<:GaussianObservationModel{T, <:AbstractMatrix{T}}}, 
    FB::ForwardBackward, 
    y::AbstractMatrix{T}, 
    FS::Vector{FilterSmooth{T}}
) where {T<:Real}

    T_steps = size(y, 2)
    K = length(obs_models)

    @threads for k in 1:K
        R_chol = cholesky(Symmetric(obs_models[k].R))
        C = obs_models[k].C

        @views @inbounds for t in 1:T_steps
            FB.loglikelihoods[k, t] = -0.5 * tr(R_chol \ Q_obs(C, FS[k].E_z[:, t, 1], FS[k].E_zz[:, :, t, 1], y[:, t]))
        end
    end 
end

"""
    mstep!(slds::AbstractHMM, FS::Vector{FilterSmooth{T}}, y::AbstractMatrix{T}, FB::ForwardBackward) where {T<:Real}

Function to carry out the M-step in Expectation-Maximization algorithm for SLDS 
"""
function mstep!(
    slds::AbstractHMM,
    FS::Vector{FilterSmooth{T}}, 
    y::AbstractMatrix{T}, 
    FB::ForwardBackward
) where {T<:Real}

    K = slds.K

    # Update initial state distribution
    update_initial_state_distribution!(slds, FB)
    # Update transition matrix
    update_transition_matrix!(slds, FB)

    γ = FB.γ
    hs = exp.(γ)

    # Get initial parameters
    old_params = vec([stateparams(slds.B[k]) for k in 1:K])
    old_params = [old_params; vec([obsparams(slds.B[k]) for k in 1:K])]

    for k in 1:K
        # Update LDS parameters using the sufficient statistics stored in FilterSmooth
        weights = vec(hs[k,:])
        
        update_initial_state_mean!(slds.B[k], FS[k].E_z)
        update_initial_state_covariance!(slds.B[k], FS[k].E_z, FS[k].E_zz)
        update_A!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev)
        update_Q!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev)
        
        # Convert to multi-trial format for compatibility with existing functions
        y_3d = reshape(y, size(y)..., 1)
        update_C!(slds.B[k], FS[k].E_z, FS[k].E_zz, y_3d, weights)

        # Note: Not updating R as mentioned in original comment
        # update_R!(slds.B[k], FS[k].E_z, FS[k].E_zz, y_3d, weights)
    end

    new_params = vec([stateparams(slds.B[k]) for k in 1:K])
    new_params = [new_params; vec([obsparams(slds.B[k]) for k in 1:K])]

    # Calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end