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
Perform Kalman filtering for continuous latent states, accounting for posterior over z_t.
"""
function kalman_filter(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64}, posterior_probs)
    T = size(observations, 2) # Number of time steps
    K = slds.K                # Number of modes
    d = slds.B[1].latent_dim  # Latent state dimension

    combined_means = zeros(d, T)
    combined_covariances = [zeros(d, d) for _ in 1:T]

    mode_means = [zeros(d) for _ in 1:T, _ in 1:K]
    mode_covariances = [zeros(d, d) for _ in 1:T, _ in 1:K]

    for k in 1:K
        mode_means[1, k] .= slds.B[k].state_model.x0
        mode_covariances[1, k] .= slds.B[k].state_model.P0
    end

    for t in 2:T
        for k in 1:K
            A = slds.B[k].state_model.A
            Q = slds.B[k].state_model.Q
            C = slds.B[k].obs_model.C
            R = slds.B[k].obs_model.R

            prior_mean = A * mode_means[t-1, k]
            prior_cov = A * mode_covariances[t-1, k] * A' + Q

            innovation = observations[:, t] - C * prior_mean
            S = C * prior_cov * C' + R

            K_gain = prior_cov * C' * inv(S)
            mode_means[t, k] .= prior_mean + K_gain * innovation
            mode_covariances[t, k] .= prior_cov - K_gain * C * prior_cov
        end

        # Combine mode-conditioned estimates weighted by posterior probabilities
        combined_means[:, t] .= sum(posterior_probs[k, t] .* mode_means[t, k] for k in 1:K)
        combined_covariances[t] .= sum(posterior_probs[k, t] .* (mode_covariances[t, k] + (mode_means[t, k] - combined_means[:, t]) * (mode_means[t, k] - combined_means[:, t])') for k in 1:K)
    end

    return combined_means, combined_covariances
end

"""
Perform HMM forward-backward for discrete mode probabilities.
"""
function hmm_forward_backward(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64}, means, covariances)
    T = size(observations, 2) # Number of time steps
    K = slds.K                # Number of modes

    log_forward = zeros(K, T)
    log_backward = zeros(K, T)
    log_emission_probs = zeros(K, T)

    for t in 1:T
        for k in 1:K
            C = slds.B[k].obs_model.C
            R = slds.B[k].obs_model.R

            innovation = observations[:, t] - C * means[:, t]
            S = C * covariances[t] * C' + R

            log_emission_probs[k, t] = -0.5 * logdet(S) - 0.5 * dot(innovation, S \ innovation)
        end
    end

    log_forward[:, 1] .= log.(slds.πₖ) .+ log_emission_probs[:, 1]
    for t in 2:T
        for k in 1:K
            max_prev_log = maximum(log_forward[:, t-1])
            weighted_sum = sum(exp.(log_forward[:, t-1] .- max_prev_log) .* slds.A[:, k])
            log_forward[k, t] = log(weighted_sum) + max_prev_log + log_emission_probs[k, t]
        end
    end

    log_backward[:, end] .= 0.0
    for t in (T-1):-1:1
        for k in 1:K
            max_next_log = maximum(log_backward[:, t+1])
            weighted_sum = sum(exp.(log_backward[:, t+1] .- max_next_log) .* slds.A[k, :])
            log_backward[k, t] = log(weighted_sum) + max_next_log + log_emission_probs[k, t+1]
        end
    end

    log_posterior = log_forward .+ log_backward
    log_posterior .-= logsumexp(log_posterior, dims=1)

    return exp.(log_posterior), exp.(log_forward), exp.(log_backward)
end

"""
E-step: Perform variational inference to compute posterior probabilities.
"""
function variational_expectation(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64})
    d = slds.B[1].latent_dim
    T = size(observations, 2)

    # Initialize means and covariances with correct dimensions
    means = zeros(d, T)
    covariances = [zeros(d, d) for _ in 1:T]

    posterior_probs, forward_probs, backward_probs = hmm_forward_backward(slds, observations, means, covariances)
    combined_means, combined_covariances = kalman_filter(slds, observations, posterior_probs)
    return posterior_probs, forward_probs, backward_probs, combined_means, combined_covariances
end

"""
M-step: Update the parameters of the SLDS using the posterior probabilities.
"""
function maximization_step!(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64}, posterior_probs)
    T = size(observations, 2)
    K = slds.K

    # Update initial probabilities
    slds.πₖ .= posterior_probs[:, 1]

    # Update transition matrix
    transition_counts = zeros(K, K)
    for t in 1:(T - 1)
        transition_counts .+= posterior_probs[:, t] * posterior_probs[:, t + 1]'
    end
    slds.A .= transition_counts ./ sum(transition_counts, dims=2)
end

"""
Fit the SLDS using Variational EM.
"""
function fit_slds!(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-4)
    prev_log_likelihood = -Inf

    for iter in 1:max_iters
        # E-step
        posterior_probs, forward_probs, backward_probs, means, covariances = variational_expectation(slds, observations)

        # M-step
        maximization_step!(slds, observations, posterior_probs)

        # Compute log-likelihood
        log_likelihood = sum(log.(sum(forward_probs, dims=1)))
        println("Iteration $iter, Log Likelihood: $log_likelihood")

        # Check for convergence
        if abs(log_likelihood - prev_log_likelihood) < tol
            break
        end

        prev_log_likelihood = log_likelihood
    end

    return slds
end

#=
function weighted_kalman_update(obs::Vector{Float64}, pred_means::Vector{Vector{Float64}}, 
                                 pred_covs::Vector{Matrix{Float64}}, 
                                 C::Matrix{Float64}, R::Matrix{Float64}, gamma_t::Vector{Float64})
    # Initialize weighted updates
    updated_mean = zeros(size(pred_means[1]))
    updated_cov = zeros(size(pred_covs[1]))

    for k in 1:length(pred_means)
        # Kalman update for mode k
        obs_pred = C * pred_means[k]
        S = C * pred_covs[k] * C' + R
        K = pred_covs[k] * C' * inv(S)
        mean_k = pred_means[k] + K * (obs - obs_pred)
        cov_k = pred_covs[k] - K * C * pred_covs[k]

        # Weight by gamma_t[k]
        updated_mean += gamma_t[k] * mean_k
        updated_cov += gamma_t[k] * cov_k
    end

    return updated_mean, Symmetric(updated_cov)
end

function compute_log_likelihoods(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64})
    T = size(observations, 2)
    n_modes = slds.K
    state_dim = slds.B[1].latent_dim

    log_likelihoods = zeros(n_modes, T)

    for k in 1:n_modes
        lds = slds.B[k]
        for t in 1:T
            pred_mean = zeros(state_dim)
            pred_cov = rand(state_dim, state_dim)  # Placeholder

            obs_mean = lds.obs_model * pred_mean
            obs_cov = Symmetric(lds.obs_model * pred_cov * lds.obs_model' + rand(state_dim))

            log_likelihoods[k, t] = logpdf(MvNormal(obs_mean, obs_cov), observations[:, t])
        end
    end

    return log_likelihoods
end

function e_step(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64})
    log_likelihoods = compute_log_likelihoods(slds, observations)
    T = size(observations, 2)

    # Forward algorithm for mode probabilities
    log_alpha = zeros(slds.K, T)
    log_alpha[:, 1] .= log.(slds.πₖ) + log_likelihoods[:, 1]

    for t in 2:T
        for k in 1:slds.K
            log_alpha[k, t] = logsumexp(log_alpha[:, t-1] + log.(slds.A[:, k])) + log_likelihoods[k, t]
        end
    end

    gamma = exp.(log_alpha .- logsumexp(log_alpha, dims=1))  # Normalize in log space
    return gamma, log_alpha
end

function m_step(slds::SwitchingLinearDynamicalSystem, gamma::Matrix{Float64}, observations::Matrix{Float64})
    T = size(observations, 2)

    # Update πₖ
    slds.πₖ = gamma[:, 1]

    # Update A (transition matrix)
    A_num = zeros(slds.K, slds.K)
    A_denom = zeros(slds.K)

    for t in 2:T
        for i in 1:slds.K
            for j in 1:slds.K
                A_num[i, j] += gamma[i, t-1] * gamma[j, t]
            end
            A_denom[i] += gamma[i, t-1]
        end
    end

    for i in 1:slds.K
        slds.A[i, :] .= A_num[i, :] ./ max(A_denom[i], 1e-8)
    end

    # Update Linear Dynamical Systems
    for k in 1:slds.K
        lds = slds.B[k]
        num_A = zeros(lds.latent_dim, lds.latent_dim)
        denom_A = zeros(lds.latent_dim, lds.latent_dim)
        num_Q = zeros(lds.latent_dim, lds.latent_dim)

        for t in 2:T
            num_A += gamma[k, t] * (observations[:, t] * observations[:, t-1]')
            denom_A += gamma[k, t] * (observations[:, t-1] * observations[:, t-1]')

            residual = observations[:, t] - lds.state_model * observations[:, t-1]
            num_Q += gamma[k, t] * (residual * residual')
        end

        lds.state_model = num_A * inv(denom_A + 1e-6 * I(size(denom_A, 1)))
        lds.obs_model = num_Q / sum(gamma[k, 2:T])  # Assuming shared model logic
    end

    return slds
end

function fit!(slds::SwitchingLinearDynamicalSystem, observations::Matrix{Float64}, max_iter::Int = 10)
    elbo_values = Float64[]

    for iter in 1:max_iter
        println("Iteration $iter")

        # E-step
        gamma, log_alpha = e_step(slds, observations)

        # Compute ELBO
        elbo = sum(logsumexp(log_alpha, dims=1))
        push!(elbo_values, elbo)
        println("ELBO: $elbo")

        # M-step
        slds = m_step(slds, gamma, observations)
    end

    return slds, elbo_values
end
=#