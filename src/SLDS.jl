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

#havce a problem here becuase FB_storage is defined later and get an error
#function variational_mstep!(model::SwitchingLinearDynamicalSystem, FB_storage::ForwardBackward, y)
    # update initial state distribution
#    update_initial_state_distribution!(model, FB_storage)
    # update transition matrix
#    update_transition_matrix!(model, FB_storage)
    #need to add updates for LDSs#
#end


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

    T_step = size(y, 2)
    FB_storage = initialize_forward_backward(slds, T_step)

    # Run EM
    for i in 1:max_iter
        # E-step
        #E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml = 
        variational_expectation!(slds, y, FB_storage)
        #ml = rand(1)[1]

        # Compute and update the log-likelihood
        #log_likelihood_current = logsumexp(FB_storage.α[:, end])
        #push!(lls, log_likelihood_current)
        #if abs(log_likelihood_current - log_likelihood) < tol
            # finish!(p)
        #    return lls
        #else
        #    log_likelihood = log_likelihood_current
        #end
        # M-Step
        #mstep!(model, FB_storage, transpose_data)

        # M-step
        #for k = 1:K
            #Δparams = mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y, w)
        #end
        # Update the log-likelihood vector and parameter difference
        #push!(mls, ml)
        #push!(param_diff, Δparams)

        # Update the progress bar
        next!(prog)

        # Check convergence
        #if abs(ml - prev_ml) < tol
        #    finish!(prog)
        #    return mls, param_diff
        #end

        #prev_ml = ml
    end

    # Finish the progress bar if max_iter is reached
    finish!(prog)

    return mls, param_diff
end

function variational_expectation!(model::SwitchingLinearDynamicalSystem, y, FB_storage)

    K = model.K
    T = size(y, 2)

    γ = FB_storage.γ
    log_likelihoods = FB_storage.loglikelihoods

    #3. compute xs from hs
    hs = exp.(γ)

    #1. compute qs from xs
    @threads for k in 1:model.K
        smoothed_x, smoothed_p, _  = smooth(model.B[k], y, vec(hs[k,:]))
        emission_loglikelihoods!(k, model.B[k].obs_model, FB_storage, y, smoothed_x, smoothed_p)
    end
    
    #2. compute hs from qs
    forward!(model, FB_storage)
    backward!(model, FB_storage)
    calculate_γ!(model, FB_storage)

end


function emission_loglikelihoods!(k, model::GaussianObservationModel, FB_storage::ForwardBackward, y, x, p)
    log_likelihoods = FB_storage.loglikelihoods

    R_chol = cholesky(Symmetric(model.R))
    C = model.C

    @threads for t in 1:T
        log_likelihoods[k, t] = -0.5 * (R_chol \ y[:,t])' * y[:,t] + 
            (R_chol \ y[:,1])' * C * x[:,t] - 
            0.5 * tr((R_chol \ C)' * C * p[:,:,t])
    end

end