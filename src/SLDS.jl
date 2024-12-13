export SwitchingLinearDynamicalSystem, fit!, loglikelihood, sample

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