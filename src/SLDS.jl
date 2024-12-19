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
    #z[1] = sample(1:K, Weights(slds.πₖ))
    x[:, 1] = rand(MvNormal(zeros(state_dim), slds.B[z[1]].state_model.Q))
    y[:, 1] = rand(MvNormal(slds.B[z[1]].obs_model.C * x[:, 1], slds.B[z[1]].obs_model.R))

    for t in 2:T
        # Sample mode based on transition probabilities
        z[t] = rand(Categorical(slds.A[z[t-1], :] ./ sum(slds.A[z[t-1], :])))
        #z[t] = sample(1:K, Weights(slds.A[z[t - 1], :]))
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

    A = rand(K, K)
    A ./= sum(A, dims=2) # Normalize rows to sum to 1

    πₖ = rand(K)
    πₖ ./= sum(πₖ) # Normalize to sum to 1

    # set up the state parameters
    A2 = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)] 
    Q = Matrix(0.1 * I(d))

    x0 = [0.0; 0.0]
    P0 = Matrix(0.1 * I(d))

    # set up the observation parameters
    C = randn(p, d)
    R = Matrix(0.5 * I(p))

    B = [LinearDynamicalSystem(
        GaussianStateModel(A2, Q, x0, P0),
        GaussianObservationModel(C, R),
        d, p, fill(true, 6  )) for _ in 1:K]

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

    return mls, param_diff
end

mutable struct FilterSmooth{T<:Real}
    x_smooth::Matrix{T}
    p_smooth::Array{T, 3}
    E_z::Array{T, 3}
    E_zz::Array{T, 4}
    E_zz_prev::Array{T, 4}
end

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

    return ml_total

end


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
                yt_Rinv * C * FS[k].x_smooth[:,t] - 0.5 * tr(C_Rinv * C * FS[k].p_smooth[:,:,t])
        end
    end 

end

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
        update_initial_state_mean!(slds.B[k], FS[k].E_z, vec(hs[k,:]))
        update_initial_state_covariance!(slds.B[k], FS[k].E_z, FS[k].E_zz, vec(hs[k,:]))
        #update_A!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev, vec(hs[k,:]))
        #update_Q!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev, vec(hs[k,:]))
        #update_C!(slds.B[k], FS[k].E_z, FS[k].E_zz, y, vec(hs[k,:]))
        #update_R!(slds.B[k], FS[k].E_z, FS[k].E_zz, y, vec(hs[k,:]))
    end

    new_params = vec([stateparams(slds.B[k]) for k in 1:K])
    new_params = [new_params; vec([obsparams(slds.B[k]) for k in 1:K])]

    # calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end