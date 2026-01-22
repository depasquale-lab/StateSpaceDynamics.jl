# Poisson LDS implementations - exports handled by types.jl and gaussian.jl

"""
    loglikelihood(
        x::AbstractMatrix{U},
        plds::LinearDynamicalSystem{T,S,O},
        y::AbstractMatrix{T}
    )

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for a
single trial.

# Arguments
- `x::AbstractMatrix{T}`: The latent state variables. Dimensions: (latent_dim, tsteps)
- `lds::LinearDynamicalSystem{T,S,O}`: The Linear Dynamical System model.
- `y::AbstractMatrix{T}`: The observed data. Dimensions: (obs_dim, tsteps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not
    currently used.

# Returns
- `ll::Vector{T}`: The log-likelihood value.

# Ref
- loglikelihood(
    x::AbstractArray{T,3},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3}
)
"""
function loglikelihood(
    x::AbstractMatrix{U}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

    # Result type and setup
    R = promote_type(T, U)
    tsteps = size(y, 2)
    ll = zeros(R, tsteps)

    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)

    # Pre-compute inverses
    inv_p0 = inv(plds.state_model.P0)
    inv_Q = inv(plds.state_model.Q)

    # Get dimensions
    C = plds.obs_model.C
    A = plds.state_model.A
    x0 = plds.state_model.x0
    obs_dim, latent_dim = size(C)
    obs_tmp = Vector{eltype(x)}(undef, obs_dim)

    @views for t in 1:tsteps
        obs_tmp .= C * x[:, t] .+ d
        ll[t] += (dot(y[:, t], obs_tmp) - sum(exp, obs_tmp))
    end

    # Prior term p(x₁) goes to t = 1
    dx1 = @view(x[:, 1]) .- plds.state_model.x0
    ll[1] += -R(0.5) * dot(dx1, inv_p0 * dx1)

    # Transition terms p(xₜ|xₜ₋₁) go to their respective t (t ≥ 2)
    A = plds.state_model.A
    b = plds.state_model.b
    trans_tmp = Vector{eltype(x)}(undef, latent_dim)

    @views for t in 2:tsteps
        trans_tmp .= x[:, t] .- (A * x[:, t - 1] .+ b)
        ll[t] += -R(0.5) * dot(trans_tmp, inv_Q * trans_tmp)
    end

    return ll
end

"""
    loglikelihood(x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3})

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for multiple trials.
"""
function loglikelihood(
    x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Calculate the log-likelihood over all trials
    ll = zeros(T, size(y, 3))

    @threads for n in axes(y, 3)
        ll[n] .= sum(loglikelihood(x[:, :, n], plds, y[:, :, n]))
    end

    return sum(ll)
end

"""
    Gradient(lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T})

Calculate the gradient of the log-likelihood of a Poisson Linear Dynamical System model for a single trial.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end

    # Extract model parameters
    A, Q, b = lds.state_model.A, lds.state_model.Q, lds.state_model.b
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d (non-log space)
    d = exp.(log_d)

    # Get dimensions
    tsteps = size(y, 2)
    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    # Precompute matrix inverses
    inv_P0 = inv(P0)
    inv_Q = inv(Q)

    # Pre-allocate gradient
    grad = zeros(T, latent_dim, tsteps)

    # Pre-allocate ALL temporary vectors (reused across timesteps)
    Cx_t = Vector{T}(undef, obs_dim)           # C * x[:, t]
    exp_term = Vector{T}(undef, obs_dim)       # exp(C * x[:, t] + d)
    innovation = Vector{T}(undef, obs_dim)     # y[:, t] - exp_term
    common_term = Vector{T}(undef, latent_dim) # C' * innovation

    # Temporary vectors for state dynamics terms
    Ax_t = Vector{T}(undef, latent_dim)        # A * x[:, t]
    Ax_prev = Vector{T}(undef, latent_dim)     # A * x[:, t-1]
    state_diff = Vector{T}(undef, latent_dim)  # Various state differences
    temp_grad = Vector{T}(undef, latent_dim)   # Temporary for accumulating gradient parts

    # Calculate gradient for each time step
    @views for t in 1:tsteps
        # Compute observation term efficiently
        # temp = exp.(C * x[:, t] .+ d)
        mul!(Cx_t, C, x[:, t])                 # Cx_t = C * x[:, t]
        for i in 1:obs_dim
            exp_term[i] = exp(Cx_t[i] + d[i])  # exp_term = exp(C * x[:, t] + d)
        end

        # common_term = C' * (y[:, t] - temp)
        innovation .= y[:, t] .- exp_term      # innovation = y[:, t] - exp_term
        mul!(common_term, C', innovation)      # common_term = C' * innovation

        if t == 1
            # First time step: common_term + A' * inv_Q * (x[:, 2] - A * x[:, t]) - inv_P0 * (x[:, t] - x0)

            # Compute A * x[:, t]
            mul!(Ax_t, A, x[:, t])

            # Compute x[:, 2] - A * x[:, t]
            state_diff .= x[:, 2] .- Ax_t

            # Compute inv_Q * (x[:, 2] - A * x[:, t])
            mul!(temp_grad, inv_Q, state_diff)

            # Compute A' * inv_Q * (x[:, 2] - A * x[:, t])
            mul!(grad[:, t], A', temp_grad)

            # Add common_term
            grad[:, t] .+= common_term

            # Subtract inv_P0 * (x[:, t] - x0)
            state_diff .= x[:, t] .- x0
            mul!(temp_grad, inv_P0, state_diff)
            grad[:, t] .-= temp_grad

        elseif t == tsteps
            # Last time step: common_term - inv_Q * (x[:, t] - A * x[:, t-1])

            # Compute A * x[:, t-1]
            mul!(Ax_prev, A, x[:, t - 1])

            # Compute x[:, t] - A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev

            # Compute inv_Q * (x[:, t] - A * x[:, t-1])
            mul!(temp_grad, inv_Q, state_diff)

            # grad[:, t] = common_term - inv_Q * (...)
            grad[:, t] .= common_term .- temp_grad

        else
            # Intermediate time steps: 
            # common_term + A' * inv_Q * (x[:, t+1] - A * x[:, t]) - inv_Q * (x[:, t] - A * x[:, t-1])

            # First part: A' * inv_Q * (x[:, t+1] - A * x[:, t])
            mul!(Ax_t, A, x[:, t])                   # Ax_t = A * x[:, t]
            state_diff .= x[:, t + 1] .- Ax_t          # state_diff = x[:, t+1] - A * x[:, t]
            mul!(temp_grad, inv_Q, state_diff)       # temp_grad = inv_Q * state_diff
            mul!(grad[:, t], A', temp_grad)          # grad[:, t] = A' * temp_grad

            # Add common_term
            grad[:, t] .+= common_term

            # Second part: - inv_Q * (x[:, t] - A * x[:, t-1])
            mul!(Ax_prev, A, x[:, t - 1])              # Ax_prev = A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev         # state_diff = x[:, t] - A * x[:, t-1]
            mul!(temp_grad, inv_Q, state_diff)       # temp_grad = inv_Q * state_diff
            grad[:, t] .-= temp_grad                 # grad[:, t] -= temp_grad
        end
    end

    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T})

Calculate the Hessian matrix of the log-likelihood for a Poisson Linear Dynamical System.
"""
function Hessian(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w=ones(T, size(y, 2))
    end

    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-compute a few things
    tsteps = size(y, 2)
    inv_Q = inv(Symmetric(Q))
    inv_P0 = inv(Symmetric(P0))

    # Calculate super and sub diagonals
    H_sub_entry = inv_Q * A
    H_super_entry = permutedims(H_sub_entry)

    # Fill the super and sub diagonals
    H_sub = [H_sub_entry for _ in 1:(tsteps - 1)]
    H_super = [H_super_entry for _ in 1:(tsteps - 1)]

    λ = zeros(T, size(C, 1))
    z = similar(λ)
    poisson_tmp = Matrix{T}(undef, size(C, 2), size(C, 2))
    H_diag = [Matrix{T}(undef, size(x, 1), size(x, 1)) for _ in 1:tsteps]

    # minnimal allocation Hessian helper function
    function calculate_poisson_hess!(out::Matrix{T}, C::Matrix{T}, λ::Vector{T}) where {T}
        n, p = size(C)
        @inbounds for j in 1:p, i in 1:p
            acc = zero(T)
            for k in 1:n
                acc += C[k, i] * λ[k] * C[k, j]
            end
            out[i, j] = -acc
        end
    end

    # Pre-computed values for the Hessian
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    Q_middle = xt1_given_xt + xt_given_xt_1
    Q_first = x_t + xt1_given_xt
    Q_last = xt_given_xt_1

    @views for t in 1:tsteps
        mul!(z, C, x[:, t])  # z = C * x[:, t]
        @. λ = exp(z + d)

        if t == 1
            H_diag[t] .= Q_first
        elseif t == tsteps
            H_diag[t] .= Q_last
        else
            H_diag[t] .= Q_middle
        end

        calculate_poisson_hess!(poisson_tmp, C, λ)
        H_diag[t] .+= poisson_tmp
    end

    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)

Calculates the Q-function for the state model over multiple trials.
"""
function Q_state(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
) where {T<:Real}
    # Calculate the Q-function for the state model
    vals = zeros(T, size(E_z, 3))

    @views @threads for k in axes(E_z, 3)
        vals[k] = Q_state(
            A, b, Q, P0, x0, E_z[:, :, k], E_zz[:, :, :, k], E_zz_prev[:, :, :, k]
        )
    end

    return sum(vals)
end

"""
    Q_observation_model(C, D, log_d, E_z, E_zz, y)

Calculate the Q-function for the observation model.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractArray{U,3},
    P_smooth::AbstractArray{U,4},
    y::AbstractArray{U,3},
) where {T<:Real,U<:Real}
    obs_dim, state_dim = size(C)

    d = exp.(log_d)
    Q_val = zero(T)
    trials = size(E_z, 3)
    tsteps = size(E_z, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    CC = zeros(T, obs_dim, state_dim^2)

    for i in 1:obs_dim
        CC[i, :] .= vec(C[i, :] * C[i, :]')
    end

    @threads for k in 1:trials
        @views for t in 1:tsteps
            Ez_t = view(E_z, :, t, k)
            P_t = view(P_smooth,:,:,t,k)
            y_t = view(y, :, t, k)

            mul!(h, C, Ez_t)          # h = C * E_z[:, t, k]
            h .+= d

            ρ .= T(0.5) .* CC * vec(P_t)
            ŷ = exp.(h .+ ρ)

            Q_val += sum(y_t .* h .- ŷ)
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, E_z, p_smooth, y, weights)

Calculate the Q-function for the observation model for a single trial with optional per-timestep weights.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real}
    obs_dim, state_dim = size(C)
    d = exp.(log_d)
    Q_val = zero(T)
    tsteps = size(y, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    temp_vec = Vector{T}(undef, state_dim)

    @views for t in 1:tsteps
        wt = isnothing(weights) ? one(T) : weights[t]

        Ez_t = E_z[:, t]
        P_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # h = C * Ez_t + d
        mul!(h, C, Ez_t)
        h .+= d

        # Compute ρ[i] = 0.5 * c_i' * P_t * c_i
        for i in 1:obs_dim
            c_i = view(C, i, :)
            mul!(temp_vec, P_t, c_i)
            ρ[i] = T(0.5) * dot(c_i, temp_vec)
        end

        # Compute ŷ = exp(h + ρ) in-place, reusing ρ as ŷ
        for i in 1:obs_dim
            ρ[i] = exp(h[i] + ρ[i])
        end

        # Compute weighted sum(y_t .* h .- ŷ)
        for i in 1:obs_dim
            Q_val += wt * (y_t[i] * h[i] - ρ[i])
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, tfs, y, w)

Calculate the Q-function for the observation model across all trials using TrialFilterSmooth with optional weights.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, trials)

    @threads for k in 1:trials
        fs = tfs[k]
        weights = isnothing(w) ? nothing : w[k]
        Q_vals[k] = Q_observation_model(
            C, log_d, fs.E_z, fs.p_smooth, view(y,:,:,k), weights
        )
    end

    return sum(Q_vals)
end

"""
    Q_function(A, b, Q, C, log_d, x0, P0, E_z, E_zz, E_zz_prev, y)

Calculate the Q-function for a single trial of a Poisson Linear Dynamical System.
"""
function Q_function(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    state_q = StateSpaceDynamics.Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
    obs_q = Q_observation_model(C, log_d, E_z, p_smooth, y)
    return state_q + obs_q
end

"""
    calculate_elbo(
        plds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T, 3},
        E_zz::AbstractArray{T, 4},
        E_zz_prev::AbstractArray{T, 4},
        P_smooth::AbstractArray{T, 4},
        y::AbstractArray{T, 3},
        total_entropy::T
    ) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

Calculate the Evidence Lower Bound (ELBO) for a Poisson Linear Dynamical System (PLDS). Adds constant-free IW log-prior terms 
for `Q` and `P0` when priors are set, so the ELBO tracks the MAP objective.

# Note
Ensure that the dimensions of input arrays match the expected dimensions as described in the
arguments section.
"""
function calculate_elbo(
    plds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Set up parameters
    A, b, Q, x0, p0 = plds.state_model.A,
    plds.state_model.b, plds.state_model.Q, plds.state_model.x0,
    plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    ntrials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials (like Gaussian version)
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            A,
            b,
            Q,
            C,
            log_d,
            x0,
            p0,
            fs.E_z,
            fs.E_zz,
            fs.E_zz_prev,
            fs.p_smooth,
            view(y,:,:,trial),
        )
    end

    # IW priors on state covariances (if present)
    prior_term = zero(T)
    if plds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.Q, plds.state_model.Q_prior)
    end

    if plds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.P0, plds.state_model.P0_prior)
    end

    return sum(Q_vals) + prior_term - total_entropy
end

"""
    gradient_observation_model_single_trial!(grad, C, log_d, E_z, p_smooth, y, weights)

Compute the gradient for a single trial and add it to the accumulated gradient.
"""
function gradient_observation_model_single_trial!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real}
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    tsteps = size(y, 2)

    # Pre-allocate temporary arrays
    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    λ = Vector{T}(undef, obs_dim)
    CP = Matrix{T}(undef, obs_dim, latent_dim)

    @views for t in 1:tsteps
        weight = isnothing(weights) ? one(T) : weights[t]

        E_z_t = E_z[:, t]
        P_smooth_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # Compute h = C * z_t + d
        mul!(h, C, E_z_t)
        h .+= d

        # Pre-compute CP = C * P_smooth_t
        mul!(CP, C, P_smooth_t)

        # Compute ρ efficiently 
        for i in 1:obs_dim
            ρ[i] = T(0.5) * dot(C[i, :], CP[i, :])
        end

        # Compute λ = exp(h + ρ)
        for i in 1:obs_dim
            λ[i] = exp(h[i] + ρ[i])
        end

        # Gradient computation with weight
        for j in 1:latent_dim
            for i in 1:obs_dim
                idx = (j - 1) * obs_dim + i
                grad[idx] += weight * (y_t[i] * E_z_t[j] - λ[i] * (E_z_t[j] + CP[i, j]))
            end
        end

        # Update log_d gradient
        @views grad[(end - obs_dim + 1):end] .+= weight .* (y_t .- λ) .* d
    end
end

"""
    gradient_observation_model!(grad, C, log_d, tfs, y, w)

Compute the gradient of the Q-function with respect to the observation model parameters using TrialFilterSmooth.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)

    fill!(grad, zero(T))

    # Accumulate gradients from all trials
    @threads for k in 1:trials
        fs = tfs[k]
        weights = isnothing(w) ? nothing : w[k]

        # Local gradient for this trial
        local_grad = zeros(T, length(grad))

        gradient_observation_model_single_trial!(
            local_grad, C, log_d, fs.E_z, fs.p_smooth, view(y,:,:,k), weights
        )

        # Thread-safe update of global gradient
        grad .+= local_grad
    end

    return grad .*= -1
end

"""
    update_observation_model!(plds, tfs, y, w)

Update the observation model parameters of a PLDS model.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if plds.fit_bool[5]
        params = vcat(vec(plds.obs_model.C), plds.obs_model.log_d)

        function f(params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return -Q_observation_model(C, log_d, tfs, y, w)
        end

        function g!(grad::Vector{T}, params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return gradient_observation_model!(grad, C, log_d, tfs, y, w)
        end

        opts = Optim.Options(;
            x_reltol=1e-12, x_abstol=1e-12, g_abstol=1e-12, f_reltol=1e-12, f_abstol=1e-12
        )

        result = optimize(
            f, g!, params, LBFGS(; linesearch=LineSearches.HagerZhang()), opts
        )

        # Update the parameters
        C_size = plds.obs_dim * plds.latent_dim
        plds.obs_model.C = reshape(
            result.minimizer[1:C_size], plds.obs_dim, plds.latent_dim
        )
        plds.obs_model.log_d = result.minimizer[(end - plds.obs_dim + 1):end]
    end

    return nothing
end

"""
    mstep!(plds, tfs, y, w)

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with multi-trial data.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Get old params
    old_params = _get_all_params_vec(plds)

    # Update state parameters
    update_initial_state_mean!(plds, tfs, w)
    update_initial_state_covariance!(plds, tfs, w)
    update_A_b!(plds, tfs, w)
    update_Q!(plds, tfs, w)

    # Update observation parameters
    update_observation_model!(plds, tfs, y, w)

    # Return parameter delta
    new_params = _get_all_params_vec(plds)
    return norm(new_params - old_params)
end
