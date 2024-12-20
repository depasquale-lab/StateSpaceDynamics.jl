export SwitchingLinearDynamicalSystem, fit!, sample, initialize_slds, variational_expectation!

"""
Switching Linear Dynamical System
"""
mutable struct SwitchingLinearDynamicalSystem
    A::Matrix{<:Real}                 # Transition matrix for mode switching
    B::Vector{LinearDynamicalSystem}  # Vector of Linear Dynamical System models
    πₖ::Vector{Float64}               # Initial state distribution
    K::Int                            # Number of modes
end



""""
    FilterSmooth{T<:Real}

A mutable structure for storing smoothed estimates and associated covariance matrices in a filtering or smoothing algorithm.

# Type Parameters
- `T<:Real`: The numerical type used for all fields (e.g., `Float64`, `Float32`).

# Fields
- `x_smooth::Matrix{T}`  
  The matrix containing smoothed state estimates over time. Each column typically represents the state vector at a given time step.

- `p_smooth::Array{T, 3}`  
  The posterior covariance matrices with dimensions (latent_dim, latent_dim, time_steps)

- `E_z::Array{T, 3}`  
  The expected latent states, size (state_dim, T, n_trials).

- `E_zz::Array{T, 4}`  
  The expected value of z_t * z_t', size (state_dim, state_dim, T, n_trials).

- `E_zz_prev::Array{T, 4}`  
  The expected value of z_t * z_{t-1}', size (state_dim, state_dim, T, n_trials).

# Example
```julia
# Initialize a FilterSmooth object with Float64 type
filter = FilterSmooth{Float64}(
    x_smooth = zeros(10, 100),
    p_smooth = zeros(10, 10, 100),
    E_z = zeros(10, 5, 100),
    E_zz = zeros(10, 10, 5, 100),
    E_zz_prev = zeros(10, 10, 5, 100)
)
"""
mutable struct FilterSmooth{T<:Real}
    x_smooth::Matrix{T}
    p_smooth::Array{T, 3}
    E_z::Array{T, 3}
    E_zz::Array{T, 4}
    E_zz_prev::Array{T, 4}
end


"""
    initialize_FilterSmooth(model::LinearDynamicalSystem, num_obs::Int) -> FilterSmooth{T}

Initialize a `FilterSmooth` object for a given linear dynamical system model and number of observations.

# Arguments
- `model::LinearDynamicalSystem`:  
  The linear dynamical system model containing system parameters, including the latent dimensionality (`latent_dim`).

- `num_obs::Int`:  
  The number of observations (time steps) for which to initialize the smoothing filters.

# Returns
- `FilterSmooth{T}`:  
  A `FilterSmooth` instance with all fields initialized to zero arrays. The dimensions of the arrays are determined by the number of states (`latent_dim`) from the model and the specified number of observations (`num_obs`).

# Example
```julia
# Assume `model` is an instance of LinearDynamicalSystem with latent_dim = 10
num_observations = 100
filter_smooth = initialize_FilterSmooth(model, num_observations)

# `filter_smooth` now contains zero-initialized arrays for smoothing operations
"""
function initialize_FilterSmooth(model::LinearDynamicalSystem, num_obs::Int)
    num_states = model.latent_dim
    FilterSmooth(
        zeros(num_states, num_obs),
        zeros(num_states, num_states, num_obs),
        zeros(num_states, num_obs, 1),
        zeros(num_states, num_states, num_obs, 1),
    zeros(num_states, num_states, num_obs, 1)
    )
end


"""
Generate synthetic data with switching LDS models
"""
function sample(slds, T::Int)
    state_dim = slds.B[1].latent_dim
    obs_dim = slds.B[1].obs_dim
    K = slds.K

    x = zeros(state_dim, T)  # Latent states
    y = zeros(obs_dim, T)   # Observations
    z = zeros(Int, T)       # Mode sequence

    # Sample initial mode
    z[1] = rand(Categorical(slds.πₖ / sum(slds.πₖ)))
    x[:, 1] = rand(MvNormal(zeros(state_dim), slds.B[z[1]].state_model.Q))
    y[:, 1] = rand(MvNormal(slds.B[z[1]].obs_model.C * x[:, 1], slds.B[z[1]].obs_model.R))

    for t in 2:T
        # Sample mode based on transition probabilities
        z[t] = rand(Categorical(slds.A[z[t-1], :] ./ sum(slds.A[z[t-1], :])))
        # Update latent state and observation
        x[:, t] = rand(MvNormal(slds.B[z[t]].state_model.A * x[:, t-1], slds.B[z[t]].state_model.Q))
        y[:, t] = rand(MvNormal(slds.B[z[t]].obs_model.C * x[:, t], slds.B[z[t]].obs_model.R))
    end

    return x, y, z
    
end


"""
Initialize a Switching Linear Dynamical System with random parameters.
"""
function initialize_slds(;K::Int=2, d::Int=2, p::Int=10, seed::Int=42)
    Random.seed!(seed)

    #A = rand(K, K)
    A = zeros(K,K)
    A[1,1] = 0.96
    A[1,2] = 0.04
    A[2,2] = 0.96
    A[2,1] = 0.04
    A ./= sum(A, dims=2) # Normalize rows to sum to 1

    πₖ = rand(K)
    πₖ ./= sum(πₖ) # Normalize to sum to 1

    # set up the state parameters
    #A2 = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)] 
    Q = Matrix(0.001 * I(d))

    x0 = [0.0; 0.0]
    P0 = Matrix(0.001 * I(d))

    # set up the observation parameters
    C = randn(p, d)
    R = Matrix(0.001 * I(p))

    B = [LinearDynamicalSystem(
        GaussianStateModel(0.95 * [cos(f) -sin(f); sin(f) cos(f)], Q, x0, P0),
        GaussianObservationModel(C, R),
        d, p, fill(true, 6  )) for (_,f) in zip(1:K, [0.1, 2.0])]

    return SwitchingLinearDynamicalSystem(A, B, πₖ, K)

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
"""
function fit!(
    slds::SwitchingLinearDynamicalSystem, y::Matrix{T}; max_iter::Int=1000, tol::Real=1e-12
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
    FB = initialize_forward_backward(slds, T_step)
    #γ = FB.γ
    #γ .= -100.
    FS = [initialize_FilterSmooth(slds.B[k], T_step) for k in 1:K]

    # Run EM
    for i in 1:max_iter
        # E-step
        ml = variational_expectation!(slds, y, FB, FS)       

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
            return mls, param_diff
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

# Arguments
- `model::SwitchingLinearDynamicalSystem`:  
  The switching linear dynamical system model containing parameters such as the number of regimes (`K`), system matrices (`B`), and observation models.

- `y`:  
  The observation data, typically a matrix where each column represents an observation at a specific time step.

- `FB`:  
  The forward-backward object that holds variables related to the forward and backward passes, including responsibilities (`γ`).

- `FS`:  
  An array of `FilterSmooth` objects, one for each regime, storing smoothed state estimates and covariances.

# Returns
- `Float64`:  
  The total Evidence Lower Bound (ELBO) computed over all regimes and observations.

# Description
This function performs the variational expectation step for a Switching Linear Dynamical System by executing the following operations:

1. **Extract Responsibilities**:  
   Retrieves the responsibilities (`γ`) from the forward-backward object and computes their exponentials (`hs`).

2. **Parallel Smoothing and Sufficient Statistics Calculation**:  
   For each regime `k` from `1` to `model.K`, the function:
   - Performs smoothing using the `smooth` function to obtain smoothed states (`x_smooth`), covariances (`p_smooth`), inverse off-diagonal terms, and total entropy.
   - Computes sufficient statistics (`E_z`, `E_zz`, `E_zz_prev`) from the smoothed estimates.
   - Calculates the ELBO contribution for the current regime and accumulates it into `ml_total`.

3. **Update Variational Distributions**:  
   - Computes the variational distributions (`qs`) from the smoothed states, which are stored as log-likelihoods in `FB`.
   - Executes the forward and backward passes to update the responsibilities (`γ`) based on the new `qs`.
   - Recalculates the responsibilities (`γ`) to reflect the updated variational distributions.

4. **Return ELBO**:  
   Returns the accumulated ELBO (`ml_total`), which quantifies the quality of the variational approximation.

# Example
```julia
# Assume `model` is an instance of SwitchingLinearDynamicalSystem with K regimes
# `y` is the observation matrix of size (num_features, num_time_steps)
# `FB` is a pre-initialized ForwardBackward object
# `FS` is an array of FilterSmooth objects, one for each regime

elbo = variational_expectation!(model, y, FB, FS)
println("Computed ELBO: ", elbo)
"""
#havce a problem here becuase FB is defined later and get an error
function variational_expectation!(model::SwitchingLinearDynamicalSystem, y, FB, FS)

    γ = FB.γ
    hs = exp.(γ)
    ml_total = 0.

    @threads for k in 1:model.K
        #3. compute xs from hs
        FS[k].x_smooth, FS[k].p_smooth, inverse_offdiag, total_entropy  = smooth(model.B[k], y, vec(hs[k,:]))
        FS[k].E_z, FS[k].E_zz, FS[k].E_zz_prev = 
            sufficient_statistics(reshape(FS[k].x_smooth, size(FS[k].x_smooth)..., 1), 
            reshape(FS[k].p_smooth, size(FS[k].p_smooth)..., 1), 
            reshape(inverse_offdiag, size(inverse_offdiag)..., 1))
        # calculate elbo
        ml_total += calculate_elbo(model.B[k], FS[k].E_z, FS[k].E_zz, FS[k].E_zz_prev, 
            reshape(FS[k].p_smooth, size(FS[k].p_smooth)..., 1), reshape(y, size(y)...,1), total_entropy)

    end
    #1. compute qs from xs, which will live as log_likelihoods in FB
    variational_qs!([model.obs_model for model in model.B], FB, y, FS)
    
    #2. compute hs from qs, which will live as γ in FB
    forward!(model, FB)
    backward!(model, FB)
    calculate_γ!(model, FB)

    ml_total += hmm_elbo(model, FB)

    return ml_total

end


"""
"""
function hmm_elbo(model::SwitchingLinearDynamicalSystem, FB)

  # Extract necessary data
  γ = FB.γ
  ξ = FB.ξ
  loglikelihoods = FB.loglikelihoods
  A = model.A   # Transition matrix
  πₖ = model.πₖ # Initial state distribution
  time_steps = size(loglikelihoods, 2)

  # Initial state probabilities
  log_p_x_z = sum(exp.(γ[:, 1]) .* log.(πₖ))

  # Transition term using ξ
  for t in 1:(time_steps - 1)
    log_p_x_z += sum(exp.(ξ[:, :, t]) .* log.(A))
  end

  # Emission term
  log_p_x_z += sum(exp.(γ) .* loglikelihoods)

  # 2. Compute log q(z)
  log_q_z = sum(exp.(γ) .* γ)

  log_p_x_z - log_q_z
end


"""
    variational_qs!(model::Vector{GaussianObservationModel{T}}, FB, y, FS) where {T<:Real}

Compute the variational distributions (`qs`) and update the log-likelihoods for a set of Gaussian observation models within a Forward-Backward framework.

# Arguments
- `model::Vector{GaussianObservationModel{T}}`  
  A vector of Gaussian observation models, where each model defines the parameters for a specific regime or state in a Switching Linear Dynamical System. Each `GaussianObservationModel` should contain fields such as the observation matrix `C` and the observation noise covariance `R`.

- `FB`  
  The Forward-Backward object that holds variables related to the forward and backward passes of the algorithm. It must contain a mutable field `loglikelihoods`, which is a matrix where each entry `loglikelihoods[k, t]` corresponds to the log-likelihood of the observation at time `t` under regime `k`.

- `y`  
  The observation data matrix, where each column represents an observation vector at a specific time step. The dimensions are typically `(num_features, num_time_steps)`.

- `FS`  
  An array of `FilterSmooth` objects, one for each regime, that store smoothed state estimates (`x_smooth`) and their covariances (`p_smooth`). These are used to compute the expected sufficient statistics needed for updating the variational distributions.

# Returns
- `Nothing`  
  The function performs in-place updates on the `FB.loglikelihoods` matrix. It does not return any value.

# Description
`variational_qs!` updates the log-likelihoods for each Gaussian observation model across all time steps based on the current smoothed state estimates. This is a critical step in variational inference algorithms for Switching Linear Dynamical Systems, where the goal is to approximate the posterior distributions over latent variables.

The function operates as follows:

1. **Initialization**:  
   - Extracts the `loglikelihoods` matrix from the `FB` object.
   - Determines the number of regimes (`K`) and the number of time steps (`T_steps`) from the `model` and observation matrix `y`.

2. **Parallel Computation Across Regimes**:  
   - Utilizes multi-threading (`@threads`) to iterate over each regime `k` in parallel.
   - For each regime:
     - Computes the Cholesky decomposition of the observation noise covariance matrix `R`.
     - Precomputes `C_Rinv`, which is used in the log-likelihood calculation to improve computational efficiency.

3. **Parallel Computation Across Time Steps**:  
   - Within each regime, another level of multi-threading (`@threads`) iterates over each time step `t`.
   - For each time step:
     - Computes the transformed observation `yt_Rinv`.
     - Calculates the log-likelihood `log_likelihoods[k, t]`
     - Updates the `log_likelihoods` matrix with the computed value.

# Example
```julia
# Define Gaussian observation models for each regime
model = [
    GaussianObservationModel(C = randn(5, 10), R = Matrix{Float64}(I, 5, 5)),
    GaussianObservationModel(C = randn(5, 10), R = Matrix{Float64}(I, 5, 5))
]

# Initialize ForwardBackward object with a preallocated loglikelihoods matrix
FB = ForwardBackward(loglikelihoods = zeros(Float64, length(model), 100))

# Generate synthetic observation data (5 features, 100 time steps)
y = randn(5, 100)

# Initialize FilterSmooth objects for each regime
FS = [
    initialize_FilterSmooth(model[k], size(y, 2)) for k in 1:length(model)
]

# Compute variational distributions and update log-likelihoods
variational_qs!(model, FB, y, FS)

# Access the updated log-likelihoods
println(FB.loglikelihoods)
"""
function variational_qs!(model::Vector{GaussianObservationModel{T}}, FB, y, FS
    ) where {T<:Real}
    log_likelihoods = FB.loglikelihoods
    K = length(model)
    T_steps = size(y, 2)

    @threads for k in 1:K

        R_chol = cholesky(Symmetric(model[k].R))
        C = model[k].C
        C_Rinv = (R_chol \ C)'

        @threads for t in 1:T_steps
            yt_Rinv = (R_chol \ y[:,t])'
            log_likelihoods[k, t] = -0.5 * yt_Rinv * y[:,t] + 
                yt_Rinv * C * FS[k].E_z[:,t,1] - 0.5 * tr(C_Rinv * C * FS[k].E_zz[:,:,t,1])
        end

        # Subtract max for numerical stability
        log_likelihoods[k, :] .-= maximum(log_likelihoods[k, :])

    end 

    # Convert to likelihoods, normalize, and back to log space
    likelihoods = exp.(log_likelihoods)
    normalized_probs = likelihoods ./ sum(likelihoods)
    log_likelihoods = log.(normalized_probs)

end


"""
"""
function mstep!(slds::SwitchingLinearDynamicalSystem,
    FS, y::Matrix{T}, FB) where {T<:Real}

    K = slds.K

    #update initial state distribution
    update_initial_state_distribution!(slds, FB)
    #update transition matrix
    update_transition_matrix!(slds, FB)

    γ = FB.γ
    hs = exp.(γ)

    # get initial parameters
    old_params = vec([stateparams(slds.B[k]) for k in 1:K])
    old_params = [old_params; vec([obsparams(slds.B[k]) for k in 1:K])]

    for k in 1:K
        # Update LDS parameters
        update_initial_state_mean!(slds.B[k], FS[k].E_z)
        update_initial_state_covariance!(slds.B[k], FS[k].E_z, FS[k].E_zz)
        update_A!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev)
        update_Q!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev)
        update_C!(slds.B[k], FS[k].E_z, FS[k].E_zz, reshape(y, size(y)...,1), vec(hs[k,:]))
        #update_R!(slds.B[k], FS[k].E_z, FS[k].E_zz, reshape(y, size(y)...,1), vec(hs[k,:]))
    end

    new_params = vec([stateparams(slds.B[k]) for k in 1:K])
    new_params = [new_params; vec([obsparams(slds.B[k]) for k in 1:K])]

    # calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end