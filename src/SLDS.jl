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
        Δparams = mstep!(slds, E_z, E_zz, E_zz_prev, p_smooth, y, FB_storage, w)
    
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

##need a posteror x ss storage thing, like FB storage
#havce a problem here becuase FB_storage is defined later and get an error

function variational_expectation!(model::SwitchingLinearDynamicalSystem, y, FB_storage)

    K = model.K
    T = size(y, 2)
    γ = FB_storage.γ
    hs = exp.(γ)
    log_likelihoods = FB_storage.loglikelihoods

    @threads for k in 1:model.K
        #3. compute xs from hs
        x_smooth, p_smooth, inverse_offdiag, total_entropy  = smooth(model.B[k], y, vec(hs[k,:]))
        E_z, E_zz, E_zz_prev = sufficient_statistics(x_smooth, p_smooth, inverse_offdiag)
        # calculate elbo
        ml_total = calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, total_entropy)
        #need x_smooth, p_smooth, E_z, E_zz, E_zz_prev

    end
    #1. compute qs from xs, which will live as log_likelihoods in FB_storage
    variational_qs!([model.obs_model for model in model.B], FB_storage, y, smoothed_x, smoothed_p)
    
    #2. compute hs from qs, which will live as γ in FB_storage
    forward!(model, FB_storage)
    backward!(model, FB_storage)
    calculate_γ!(model, FB_storage)

end


function variational_qs!(model::Vector{GaussianObservationModel}, FB_storage, y, x, p)
    log_likelihoods = FB_storage.loglikelihoods

    @threads for k in 1:model.K

        R_chol = cholesky(Symmetric(model[k].R))
        C = model[k].C
        C_Rinv = (R_chol \ C)'

        @threads for t in 1:T
            yt_Rinv = (R_chol \ y[:,t])'
            log_likelihoods[k, t] = -0.5 * yt_Rinv * y[:,t] + 
                yt_Rinv * C * x[:,t] - 0.5 * tr(C_Rinv * C * p[:,:,t])
        end
    end 

end

function mstep!(
    slds::SwitchingLinearDynamicalSystem{S,O},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
    p_smooth::Array{T,4},
    y::Array{T,3}, FB_storage, 
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real}
    # get initial parameters
    old_params = vec(stateparams(lds))
    old_params = [old_params; vec(obsparams(lds))]

    #update initial state distribution
    update_initial_state_distribution!(model, FB_storage)
    #update transition matrix
    update_transition_matrix!(model, FB_storage)

    # Update LDS parameters
    update_initial_state_mean!(lds, E_z, w)
    update_initial_state_covariance!(lds, E_z, E_zz, w)
    update_A!(lds, E_zz, E_zz_prev)
    update_Q!(lds, E_zz, E_zz_prev)
    update_C!(lds, E_z, E_zz, y)
    update_R!(lds, E_z, E_zz, y)

    # get new params
    new_params = vec(stateparams(lds))
    new_params = [new_params; vec(obsparams(lds))]

    # calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end