export SwitchingLinearDynamicalSystem, fit!, loglikelihood, sample, variational_expectation

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

function compute_variational_h(model::SwitchingLinearDynamicalSystem, y)

    T_steps = size(y, 2)
    FB_storage = initialize_forward_backward(model, T_steps)
    log_likelihoods = FB_storage.loglikelihoods

    # Calculate observation wise likelihoods for all states
    @threads for k in 1:model.K
        #log_likelihoods[k, :] .= NEED TO SET to q
    end
    
    forward!(model, FB_storage)
    backward!(model, FB_storage)
    exp.(calculate_γ!(model, FB_storage))

end

#1. compute qs from xs
#2. compute hs from qs
#3. xs = [smooth(true_ssm.B[k], observations, hs) for i in k:K] xs from hs

#=
# Assuming you have defined the following:
# - initialize_LDS_parameters, initialize_switching_process
# - forward_backward, kalman_smoothing
# - update_LDS_parameters, update_switching_parameters
# - compute_bound (optional for monitoring convergence)

# Initialize model parameters
A, C, Q, R = initialize_LDS_parameters(M, K, D)
x0, P0 = initialize_state_priors(M, K)
pi, beta = initialize_switching_process(M)

# Initialize responsibilities and q
h = zeros(M, T)
q_vals = zeros(M, T)

old_bound = -Inf
tol_outer = 1e-5
max_iter_outer = 100

for outer_iter in 1:max_iter_outer
    # === E-Step ===
    # Compute q[m, t] based on prediction errors (use your specific method)
    for m in 1:M
        for t in 1:T
            # Compute prediction error for state transition
            if t > 1
                pred_error = x[m, t] - A[m, :] * x[m, t-1]
                q_vals[m, t] = exp(-0.5 * sum(abs2, Q_chol \ pred_error))
            else
                # Initial state contribution
                init_error = x[m, t] - x0[m]
                q_vals[m, t] = exp(-0.5 * sum(abs2, P0_chol \ init_error))
            end

            # Compute prediction error for observation
            obs_error = y[:, t] - C[m, :] * x[m, t]
            q_vals[m, t] *= exp(-0.5 * sum(abs2, R_chol \ obs_error))
        end
    end

    # Compute h[m, t] using forward-backward with q_vals
    h = forward_backward(pi, beta, q_vals)  # h should be M x T

    # Run Kalman smoothing for each model
    for m in 1:M
        # Extract model-specific parameters
        A_m = A[m, :]
        C_m = C[m, :]
        Q_m = Q[m, :]
        # Run Kalman smoother with responsibilities h[m, :]
        x_smooth_m, V_smooth_m = kalman_smoothing(A[m, :], C[m, :], Q[m, :], R, y, h[m, :], x0[m], P0[m, :])
        # Store or update state estimates as needed
        # e.g., x[m, :] = x_smooth_m
    end

    # === M-Step ===
    # Update LDS parameters using h[m, t] and smoothed states
    for m in 1:M
        # Update A[m], C[m], Q[m], x0[m], P0[m] based on h[m, :] and x_smooth_m, V_smooth_m
        A[m, :], C[m, :], Q[m, :], x0[m], P0[m, :] = update_LDS_parameters(y, h[m, :], x_smooth_m, V_smooth_m)
    end

    # Update switching process parameters pi and beta using h[m, t]
    pi, beta = update_switching_parameters(h)

    # === Compute Log-Likelihood (optional) ===
    # Compute the weighted log-likelihood for monitoring
    current_ll = 0.0
    for m in 1:M
        current_ll += weighted_loglikelihood(x[m, :], lds, y, h[m, :])
    end

    # Check for convergence
    if abs(current_ll - old_bound) < tol_outer
        println("Converged after $outer_iter iterations with log-likelihood: $current_ll")
        break
    end
    old_bound = current_ll
end

println("Training completed.")
=#