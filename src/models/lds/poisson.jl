# Poisson LDS implementations - exports handled by types.jl and gaussian.jl

"""
    joint_loglikelihood(
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
- joint_loglikelihood(
    x::AbstractArray{T,3},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3}
)
"""
function joint_loglikelihood(
    x::AbstractMatrix{U}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

    # Result type and setup
    R = promote_type(T, U)
    tsteps = size(y, 2)
    ll = zeros(R, tsteps)

    # Canonical Poisson GLM: λ_t = exp(C x_t + d), where `d` is the
    # log-link intercept (free in ℝ; positivity is provided by exp).
    d = plds.obs_model.d

    # Pre-compute Cholesky factorizations
    P0_chol = cholesky(Symmetric(plds.state_model.P0))
    Q_chol = cholesky(Symmetric(plds.state_model.Q))

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
    ll[1] += -R(0.5) * dot(dx1, P0_chol \ dx1)

    # Transition terms p(xₜ|xₜ₋₁) go to their respective t (t ≥ 2)
    A = plds.state_model.A
    b = plds.state_model.b
    trans_tmp = Vector{eltype(x)}(undef, latent_dim)

    @views for t in 2:tsteps
        trans_tmp .= x[:, t] .- (A * x[:, t - 1] .+ b)
        ll[t] += -R(0.5) * dot(trans_tmp, Q_chol \ trans_tmp)
    end

    return ll
end

"""
    _loglikelihood_ws(x, lds, y, ws)

Efficient log-likelihood computation using cached Cholesky factors from SmoothWorkspace.
Returns the total log-likelihood (scalar), not per-timestep.
Used in the Newton line search to avoid repeated Cholesky factorizations.
"""
function _loglikelihood_ws(
    x::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(y, 2)

    C = lds.obs_model.C
    d = lds.obs_model.d
    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0

    ll = zero(T)

    η = ws.temp_dy                      # length obs_dim
    dx = ws.temp_dx                     # length latent_dim
    z = ws.temp_solve_Q                # length latent_dim

    # Observation term: sum_t [ y_t'η_t - sum(exp(η_t)) ], η_t = Cx_t + d
    @views for t in 1:tsteps
        mul!(η, C, x[:, t])             # η := C * x_t
        @. η = η + d                    # η := η + d
        ll += dot(y[:, t], η) - sum(exp, η)
    end

    # Bind the raw Cholesky factor matrices once and use `LAPACK.trtrs!`
    # directly. `pdm.chol.L` would allocate a fresh
    # `LowerTriangular{T,Matrix{T}}` wrapper on every access; PDMats
    # stores the upper factor in `.chol.factors` (uplo='U'), so trans='T'
    # turns the call into a solve against L = U'.
    P0_factors = ws.P0_PD[].chol.factors
    Q_factors = ws.Q_PD[].chol.factors

    # Prior: -0.5 * || P0^{-1/2} (x1 - x0) ||^2  with P0 = U'U
    @views begin
        @. dx = x[:, 1] - x0
        copyto!(z, dx)
        LinearAlgebra.LAPACK.trtrs!('U', 'T', 'N', P0_factors, z)   # z := L \ dx
        ll -= T(0.5) * dot(z, z)
    end

    # Transitions: -0.5 * sum_{t=2}^T || Q^{-1/2} (x_t - A x_{t-1} - b) ||^2, Q = U'U
    @views for t in 2:tsteps
        mul!(dx, A, x[:, t - 1])          # dx := A * x_{t-1}
        @. dx = x[:, t] - (dx + b)      # dx := x_t - (A x_{t-1} + b)
        copyto!(z, dx)
        LinearAlgebra.LAPACK.trtrs!('U', 'T', 'N', Q_factors, z)    # z := L \ dx
        ll -= T(0.5) * dot(z, z)
    end

    return ll
end

function joint_loglikelihood!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::LDSConstantCache{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(y, 2)
    @assert length(ll) == tsteps

    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0

    # Poisson obs: λ = exp(Cx + d); log p(y|x) = sum(y .* logλ - λ - log(y!))
    z = ws.z
    λ = ws.λ

    Q_U = cc.Q_PD[].chol.U
    P0_U = cc.P0_PD[].chol.U

    dxt = ws.dxt
    tmp = ws.tmp1

    C = cc.C
    d = cc.d

    for t in 1:tsteps
        ll_t = zero(T)

        # obs: z = Cx + d ; λ = exp(z)
        @views mul!(z, C, x[:, t])
        z .+= d
        @. λ = exp(z)

        # compute y⋅z - λ - log(y!)  (loggamma(n+1) = log(n!) for real n≥0)
        @views begin
            ll_t += sum(y[:, t] .* z) - sum(λ) - sum(yi -> loggamma(yi + one(T)), y[:, t])
        end

        if t == 1
            @views dxt .= x[:, 1] .- x0
            ldiv!(dxt, P0_U, dxt)
            ll_t += cc.cP0 - T(0.5) * sum(abs2, dxt)
        else
            @views mul!(tmp, A, x[:, t - 1])
            @views tmp .= x[:, t] .- tmp .- b
            ldiv!(tmp, Q_U, tmp)
            ll_t += cc.cQ - T(0.5) * sum(abs2, tmp)
        end

        ll[t] = ll_t
    end

    return ll
end

"""
    joint_loglikelihood(x, plds, y)

Multi-trial complete-data log-likelihood for a Poisson LDS. `x` and `y` are vectors
of per-trial matrices.
"""
function joint_loglikelihood(
    x::AbstractVector{<:AbstractMatrix{<:Real}},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    ntrials = length(y)
    chunks = collect(partition(1:ntrials, max(1, cld(ntrials, Threads.nthreads()))))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            acc = zero(T)
            for n in chunk
                acc += sum(joint_loglikelihood(x[n], plds, y[n]))
            end
            acc
        end
    end
    return sum(fetch.(tasks))
end

"""
    loglikelihood(plds, y)

Marginal (observed-data) log-likelihood for a Poisson LDS — **not implemented**.

The marginal `log p(y) = ∫ p(x, y) dx` is intractable for the Poisson observation
model (non-conjugate; there is no closed-form Kalman filter as in the Gaussian case).
Use `joint_loglikelihood(x, plds, y)` for the complete-data log-likelihood given a
trajectory `x`, or the ELBO returned by `fit!` as a lower bound on `log p(y)`.
"""
function loglikelihood(
    plds::LinearDynamicalSystem{T,S,O}, y
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    return error(
        "marginal loglikelihood is not implemented for the Poisson LDS (the marginal " *
        "log p(y) is intractable). Use joint_loglikelihood(x, plds, y) for the " *
        "complete-data log-likelihood, or the ELBO from fit! as a lower bound.",
    )
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
    C, d = lds.obs_model.C, lds.obs_model.d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Get dimensions
    tsteps = size(y, 2)
    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    # Precompute Cholesky factorizations
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

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

            # Compute Q \ (x[:, 2] - A * x[:, t])
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol, temp_grad)

            # Compute A' * Q \ (x[:, 2] - A * x[:, t])
            mul!(grad[:, t], A', temp_grad)

            # Add common_term
            grad[:, t] .+= common_term

            # Subtract P0 \ (x[:, t] - x0)
            state_diff .= x[:, t] .- x0
            copyto!(temp_grad, state_diff)
            ldiv!(P0_chol, temp_grad)
            grad[:, t] .-= temp_grad

        elseif t == tsteps
            # Last time step: common_term - Q \ (x[:, t] - A * x[:, t-1])

            # Compute A * x[:, t-1]
            mul!(Ax_prev, A, x[:, t - 1])

            # Compute x[:, t] - A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev

            # Compute Q \ (x[:, t] - A * x[:, t-1])
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol, temp_grad)

            # grad[:, t] = common_term - Q \ (...)
            grad[:, t] .= common_term .- temp_grad

        else
            # Intermediate time steps:
            # common_term + A' * Q \ (x[:, t+1] - A * x[:, t]) - Q \ (x[:, t] - A * x[:, t-1])

            # First part: A' * Q \ (x[:, t+1] - A * x[:, t])
            mul!(Ax_t, A, x[:, t])                   # Ax_t = A * x[:, t]
            state_diff .= x[:, t + 1] .- Ax_t       # state_diff = x[:, t+1] - A * x[:, t]
            copyto!(temp_grad, state_diff)           # temp_grad = state_diff
            ldiv!(Q_chol, temp_grad)                 # temp_grad = Q \ state_diff
            mul!(grad[:, t], A', temp_grad)          # grad[:, t] = A' * temp_grad

            # Add common_term
            grad[:, t] .+= common_term

            # Second part: - Q \ (x[:, t] - A * x[:, t-1])
            mul!(Ax_prev, A, x[:, t - 1])           # Ax_prev = A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev        # state_diff = x[:, t] - A * x[:, t-1]
            copyto!(temp_grad, state_diff)           # temp_grad = state_diff
            ldiv!(Q_chol, temp_grad)                 # temp_grad = Q \ state_diff
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
    C, d = lds.obs_model.C, lds.obs_model.d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Pre-compute a few things
    tsteps = size(y, 2)
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    # Calculate super and sub diagonals
    H_sub_entry = Q_chol \ A
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
        for j in 1:p, i in 1:p
            acc = zero(T)
            for k in 1:n
                acc += C[k, i] * λ[k] * C[k, j]
            end
            out[i, j] = -acc
        end
    end

    # Pre-computed values for the Hessian
    state_dim = size(A, 1)
    I_mat = Matrix{T}(I, state_dim, state_dim)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

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
    Hessian!(ws, lds, y, x; w=nothing)

In-place version of `Hessian` for Poisson LDS that writes blocks into the workspace
and updates the sparse matrix values. Returns the workspace's sparse matrix.
"""
function Hessian!(
    ws::BlockTridiagonalWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end

    A, Q = lds.state_model.A, lds.state_model.Q
    C, d = lds.obs_model.C, lds.obs_model.d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    tsteps = size(y, 2)
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    H_sub_entry = Q_chol \ A
    H_super_entry = permutedims(H_sub_entry)

    for i in 1:(tsteps - 1)
        copyto!(ws.H_sub[i], H_sub_entry)
        copyto!(ws.H_super[i], H_super_entry)
    end

    state_dim = size(A, 1)
    obs_dim = size(C, 1)
    λ = zeros(T, obs_dim)
    z = similar(λ)
    poisson_tmp = Matrix{T}(undef, state_dim, state_dim)

    function calculate_poisson_hess!(out::Matrix{T}, C::Matrix{T}, λ::Vector{T}) where {T}
        n, p = size(C)
        for j in 1:p, i in 1:p
            acc = zero(T)
            for k in 1:n
                acc += C[k, i] * λ[k] * C[k, j]
            end
            out[i, j] = -acc
        end
    end

    I_mat = Matrix{T}(I, state_dim, state_dim)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

    Q_middle = xt1_given_xt + xt_given_xt_1
    Q_first = x_t + xt1_given_xt
    Q_last = xt_given_xt_1

    @views for t in 1:tsteps
        mul!(z, C, x[:, t])
        @. λ = exp(z + d)

        if t == 1
            ws.H_diag[t] .= Q_first
        elseif t == tsteps
            ws.H_diag[t] .= Q_last
        else
            ws.H_diag[t] .= Q_middle
        end

        calculate_poisson_hess!(poisson_tmp, C, λ)
        ws.H_diag[t] .+= poisson_tmp
    end

    block_tridgm!(ws)

    return ws.H_sparse
end

"""
    Q_obs!(sws, lds, E_z, p_smooth, y; weights=nothing)

Allocation-free Poisson observation-model Q for a *single trial* (full
expected complete log-likelihood, including the `-log(y!)` normalizer):

```
Q = Σ_t w_t [ y_t' h_t  -  1' exp(h_t + ρ_t)  -  Σ_i log Γ(y_{t,i} + 1) ]
```

where `h_t = C · E[x_t] + d` and `ρ_{i,t} = ½ c_i' P_t c_i`, and `d` is the
canonical log-link Poisson intercept (free in ℝ). Including the factorial
term means `calculate_elbo` matches the Laplace-approximation marginal
log-likelihood at the EM fixed point, instead of being off by a fixed
data-only constant.
"""
function Q_obs!(
    sws::SmoothWorkspace{T},                      # must provide cc.C and cc.d
    lds::LinearDynamicalSystem{T,S,O},            # for obs_model.C and obs_model.d
    E_z::AbstractMatrix{T},                       # state_dim × T
    p_smooth::AbstractArray{T,3},                 # state_dim × state_dim × T
    y::AbstractMatrix{T};                         # obs_dim × T
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    d = lds.obs_model.d
    C = lds.obs_model.C

    obs_dim, _ = size(C)
    tsteps = size(y, 2)

    # workspace buffers
    h = sws.h_obs::Vector{T}            # obs_dim
    rho = sws.rho_obs::Vector{T}            # obs_dim
    CP = sws.CP_obs::Matrix{T}            # obs_dim × state_dim
    CEz = sws.CEz_obs::Vector{T}            # obs_dim

    Q_val = zero(T)

    @views for t in 1:tsteps
        wt = isnothing(weights) ? one(T) : weights[t]

        Ez_t = E_z[:, t]                 # state_dim
        P_t = p_smooth[:, :, t]         # state_dim × state_dim
        y_t = y[:, t]                   # obs_dim

        # CEz = C * Ez_t
        mul!(CEz, C, Ez_t)

        # h = CEz + d
        @. h = CEz + d

        # CP = C * P_t
        mul!(CP, C, P_t)

        # rho[i] = 0.5 * dot(CP[i,:], C[i,:])
        for i in 1:obs_dim
            rho[i] = T(0.5) * dot(view(CP, i, :), view(C, i, :))
        end

        # rho := exp(h + rho)
        @. rho = exp(h + rho)

        log_fact = zero(T)
        for i in 1:obs_dim
            log_fact += loggamma(y_t[i] + one(T))
        end

        # Q += wt * (dot(y_t, h) - sum(rho) - log_fact)
        Q_val += wt * (dot(y_t, h) - sum(rho) - log_fact)
    end

    return Q_val
end

"""
    gradient_observation_model_single_trial!(grad, C, d, E_z, p_smooth, y, weights)

Compute the gradient for a single trial and add it to the accumulated gradient.

Treats `[C d]` as a single regression matrix `W` of size `obs_dim × (latent_dim + 1)`
on the augmented latent `z_aug = [x; 1]`. The parameter layout is unchanged from the
unstacked version — `vcat(vec(C), d) == vec([C d])` in column-major order — so callers
of this function don't need to know about the stacked view; only the gradient
computation has been unified across the two blocks. The MN-prior penalty (added by
`update_observation_model!`) lives over the same stacked `W`.
"""
function gradient_observation_model_single_trial!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}},
    h::AbstractVector{T},
    ρ::AbstractVector{T},
    λ::AbstractVector{T},
    CP::AbstractMatrix{T},
) where {T<:Real}
    obs_dim, latent_dim = size(C)
    Dp1 = latent_dim + 1
    tsteps = size(y, 2)

    # 2-D view of the gradient buffer as `[C d]`-shaped W (obs_dim × Dp1).
    grad_W = reshape(view(grad, 1:(obs_dim * Dp1)), obs_dim, Dp1)

    @views for t in 1:tsteps
        weight = isnothing(weights) ? one(T) : weights[t]

        E_z_t = E_z[:, t]
        P_smooth_t = p_smooth[:, :, t]
        y_t = y[:, t]

        mul!(h, C, E_z_t)
        h .+= d
        mul!(CP, C, P_smooth_t)

        @views for i in 1:obs_dim
            ρ[i] = T(0.5) * dot(C[i, :], CP[i, :])
            λ[i] = exp(h[i] + ρ[i])
        end

        # Stacked gradient ∂Q/∂W[i, j] = w · (y_t[i] − λ[i]) · z_aug[j]  −  w · λ[i] · CP_aug[i, j]
        # where z_aug = [E_z_t; 1] and CP_aug[:, 1:D] = CP, CP_aug[:, D+1] = 0
        # (since ρ has no dependence on the d-column of W). Split into the two
        # blocks to skip the always-zero CP_aug column.
        for j in 1:latent_dim
            for i in 1:obs_dim
                grad_W[i, j] += weight * (y_t[i] * E_z_t[j] - λ[i] * (E_z_t[j] + CP[i, j]))
            end
        end
        # j = latent_dim + 1: z_aug[j] = 1, CP_aug[:, j] = 0 → grad ← grad + w · (y − λ)
        @views grad_W[:, Dp1] .+= weight .* (y_t .- λ)
    end
end

"""
    gradient_observation_model!(grad, C, d, tfs, y, w)

Compute the gradient of the Q-function with respect to the observation model parameters using TrialFilterSmooth.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing;
    tasks_per_thread::Int=2,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    npar = length(grad)
    @assert length(sws_pool[1].CD) == npar && length(sws_pool[1].Syz) == npar "Poisson gradient accumulator size $(length(sws_pool[1].CD)) ≠ npar=$npar (obs_input_dim must be 0)"

    # Cap ntasks at `length(sws_pool)` so each task gets its own
    # pre-allocated workspace slot indexed by its position in the chunk
    # iteration (not `threadid()`, which can migrate under task
    # scheduling).
    desired = max(1, tasks_per_thread * Threads.nthreads())
    ntasks = min(trials, desired, length(sws_pool))
    chunk_size = max(1, cld(trials, ntasks))
    chunks = collect(partition(1:trials, chunk_size))

    tasks = Task[]
    @sync begin
        for (task_idx, chunk) in enumerate(chunks)
            push!(
                tasks,
                Threads.@spawn begin
                    # Each task owns one workspace from the pool. Buffers
                    # used by `gradient_observation_model_single_trial!`
                    # (h/ρ/λ/CP) come from this workspace's existing
                    # `Q_obs!` scratch fields, and the per-task `acc`/`tmp`
                    # gradient accumulators are views into `.CD` / `.Syz`
                    # (both sized `obs_dim × Dp1 = npar` for Poisson, where
                    # `obs_input_dim = 0`).
                    sws = sws_pool[task_idx]
                    acc = vec(sws.CD)
                    tmp = vec(sws.Syz)
                    fill!(acc, zero(T))

                    h_buf = sws.h_obs
                    ρ_buf = sws.rho_obs
                    λ_buf = sws.CEz_obs
                    CP_buf = sws.CP_obs

                    for k in chunk
                        fill!(tmp, zero(T))

                        fs = tfs[k]
                        weights = isnothing(w) ? nothing : w[k]

                        gradient_observation_model_single_trial!(
                            tmp,
                            C,
                            d,
                            fs.x_smooth,
                            fs.p_smooth,
                            y[k],
                            weights,
                            h_buf,
                            ρ_buf,
                            λ_buf,
                            CP_buf,
                        )

                        @simd for i in 1:npar
                            acc[i] += tmp[i]
                        end
                    end

                    return acc
                end
            )
        end
    end

    # Deterministic reduction on the caller thread
    fill!(grad, zero(T))
    for t in tasks
        acc = fetch(t)::Vector{T}  # type assertion avoids type instability from fetch
        @simd for i in 1:npar
            grad[i] += acc[i]
        end
    end

    @. grad = -grad
    return grad
end

"""
    update_observation_model!(plds, tfs, y, w)

Update the observation model parameters of a PLDS model.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    plds.fit_bool[5] || return nothing

    sws = sws_pool[1]       # f(params) is sequential; one workspace suffices
    obs_dim = plds.obs_dim
    latent_dim = plds.latent_dim
    Dp1 = latent_dim + 1
    n_W = obs_dim * Dp1     # total params; identical to vec([C d]) length

    # Param vector layout: vcat(vec(C), d) == vec([C d]) in column-major order
    params = vcat(vec(plds.obs_model.C), plds.obs_model.d)

    # MN-only prior on the stacked emission matrix W = [C d] (Poisson has no IW
    # counterpart since there is no observation-noise covariance):
    CD_prior = plds.obs_model.CD_prior

    function f(params::Vector{T})
        Cd_view = reshape(view(params, 1:n_W), obs_dim, Dp1)
        @views C_view = Cd_view[:, 1:latent_dim]
        @views d_view = Cd_view[:, Dp1]

        copyto!(plds.obs_model.C, C_view)
        copyto!(plds.obs_model.d, d_view)

        acc = zero(T)
        ntrials = length(tfs)

        for trial in 1:ntrials
            fs = tfs[trial]
            weights = isnothing(w) ? nothing : w[trial]
            # `x_smooth` is the same value as `E_z` for a Gaussian state model
            # — see `gradient_observation_model!` for context.
            acc += Q_obs!(sws, plds, fs.x_smooth, fs.p_smooth, y[trial]; weights=weights)
        end

        f_prior = zero(T)
        if CD_prior !== nothing
            Wm = Cd_view .- CD_prior.M₀
            f_prior = T(0.5) * sum(Wm .* (Wm * CD_prior.Λ))
        end

        return -acc + f_prior
    end

    function g!(grad::Vector{T}, params::Vector{T})
        Cd_view = reshape(view(params, 1:n_W), obs_dim, Dp1)
        @views C_view = Cd_view[:, 1:latent_dim]
        @views d_view = Cd_view[:, Dp1]
        gradient_observation_model!(grad, C_view, d_view, tfs, y, sws_pool, w)
        if CD_prior !== nothing
            grad_W_view = reshape(view(grad, 1:n_W), obs_dim, Dp1)
            grad_W_view .+= (Cd_view .- CD_prior.M₀) * CD_prior.Λ
        end
        return grad
    end

    opts = Optim.Options(;
        x_reltol=1e-8,
        x_abstol=1e-8,
        g_abstol=1e-8,
        f_reltol=1e-8,
        f_abstol=1e-8,
        iterations=200,
    )

    result = optimize(f, g!, params, LBFGS(; linesearch=HagerZhang()), opts)

    # surface a non-converged inner solve if applicable
    Optim.converged(result) || @warn(
        "Poisson emission M-step (LBFGS) did not converge; using last iterate",
        iterations = Optim.iterations(result),
        g_residual = Optim.g_residual(result),
    )

    # write final params back
    result_W = reshape(result.minimizer[1:n_W], obs_dim, Dp1)
    @views plds.obs_model.C .= result_W[:, 1:latent_dim]
    @views plds.obs_model.d .= result_W[:, Dp1]

    return nothing
end

"""
    mstep!(plds, suf, tfs, y, sws)

Suf-based M-step for a Poisson LDS. The Gaussian-state half (x0, P0, A&b, Q)
is updated from the aggregated sufficient statistics in `suf`. The Poisson
emission `[C d]` is non-conjugate and still goes through the existing LBFGS
routine (`update_observation_model!`) which reads `fs.x_smooth` / `fs.p_smooth`
directly from `tfs`.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    sws = sws_pool[1]
    update_initial_state_mean!(plds, suf)
    update_initial_state_covariance!(plds, suf, sws)
    update_A_b!(plds, suf, sws)
    update_Q!(plds, suf, sws)
    update_observation_model!(plds, tfs, y, sws_pool)
    return nothing
end

"""
    calculate_elbo(plds, suf, tfs, y, sws_pool)

Suf-based Poisson ELBO. Mirrors the Gaussian TD path's split:

* state-side Q-term via `Q_state!(sws, plds, suf)` from the aggregated
  sufficient statistics (O(D³) per E-step, not O(N·T·D²)),
* observation-side Q-term per-trial via the existing Poisson `Q_obs!`,
  which is irreducibly non-conjugate (no aggregator equivalent),
* posterior entropy from `tfs[trial].entropy` (filled by `smooth!`),
* `IWPrior` log-prior contributions on `Q` and `P0`, and the MN log-prior
  trace term on `[C d]` to match the LBFGS objective.
"""
function calculate_elbo(
    plds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    ntrials = length(y)

    total_entropy = zero(T)
    for fs in tfs.FilterSmooths
        total_entropy += fs.entropy
    end

    # State-side Q via aggregated suff-stats. `compute_smooth_constants!` on a
    # Poisson LDS only fills state-side constants (Q_PD / P0_PD /
    # derived blocks); `Q_state!(sws, lds, suf)` reads exactly those.
    compute_smooth_constants!(sws_pool[1], plds)
    Q_state_total = Q_state!(sws_pool[1], plds, suf)

    # Per-trial Poisson Q_obs. Each task uses its own sws (the Poisson
    # buffers `h_obs` / `rho_obs` / `CP_obs` / `CEz_obs` are local to sws).
    ntasks = min(ntrials, length(sws_pool))
    partial = zeros(T, ntasks)
    chunksize = cld(ntrials, ntasks)
    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue
        @spawn begin
            sws = sws_pool[i]
            acc = zero(T)
            for trial in lo:hi
                fs = tfs[trial]
                acc += Q_obs!(sws, plds, fs.x_smooth, fs.p_smooth, y[trial])
            end
            partial[i] = acc
        end
    end
    Q_obs_total = sum(partial)

    prior_term = zero(T)
    if plds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.Q, plds.state_model.Q_prior)
    end
    if plds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.P0, plds.state_model.P0_prior)
    end

    # MN log-prior on [C d]. No row covariance Σ for Poisson, so this is the
    # plain quadratic kernel `-½ tr((W - M₀) Λ (W - M₀)')` (matches the
    # `+½ tr(...)` penalty `update_observation_model!` adds to its LBFGS
    # objective). Λ-only and Λ-logdet constants are absorbed into the
    # additive ELBO constant.
    if plds.obs_model.CD_prior !== nothing
        D = plds.latent_dim
        W_cd = Matrix{T}(undef, plds.obs_dim, D + 1)
        @views W_cd[:, 1:D] .= plds.obs_model.C
        @views W_cd[:, D + 1] .= plds.obs_model.d
        prior = plds.obs_model.CD_prior
        Wm = W_cd .- prior.M₀
        prior_term -= T(0.5) * sum(Wm .* (Wm * prior.Λ))
    end

    return Q_state_total + Q_obs_total + prior_term + total_entropy
end

"""
    _fill_hessian_blocks_poisson!(ws, lds, x)

Fill the Hessian block diagonal and off-diagonal entries for Poisson LDS.
Uses pre-computed state model terms from `compute_smooth_constants!`
and computes the x-dependent Poisson observation term per-timestep.
"""
function _fill_hessian_blocks_poisson!(
    ws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(x, 2)
    btd = ws.btd
    C = lds.obs_model.C
    d = lds.obs_model.d
    obs_dim, latent_dim = size(C)

    # Diagonal templates (the state-side part is constant for all t; we
    # build them into pre-existing workspace scratch to avoid the two
    # `Q_middle = ... .+ ...` / `Q_first = ... .+ ...` allocations).
    Q_middle = ws.elbo_temp                              # (D × D) scratch
    Q_first = ws.elbo_temp2                              # (D × D) scratch
    @. Q_middle = ws.xt1_given_xt + ws.xt_given_xt_1
    @. Q_first = ws.x_t + ws.xt1_given_xt
    Q_last = ws.xt_given_xt_1                            # already at-rest in ws

    # Fill sub/super-diagonal blocks (constant for all timesteps)
    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], ws.H_sub_entry)
        copyto!(btd.H_super[i], ws.H_super_entry)
    end

    # Reuse existing obs-dim workspace scratch instead of allocating
    # fresh per-call `λ` / `z` vectors. `h_obs` / `rho_obs` are owned
    # by `Q_obs!`, which isn't on the Hessian-construction call path.
    λ = ws.h_obs
    z = ws.rho_obs

    @views for t in 1:tsteps
        # Start with state model contribution
        if t == 1
            copyto!(btd.H_diag[t], Q_first)
        elseif t == tsteps
            copyto!(btd.H_diag[t], Q_last)
        else
            copyto!(btd.H_diag[t], Q_middle)
        end

        # Add Poisson observation term: -C' * diag(λ) * C
        mul!(z, C, x[:, t])
        @. λ = exp(z + d)

        # Compute -C' * diag(λ) * C and add to diagonal block
        for j in 1:latent_dim, i in 1:latent_dim
            acc = zero(T)
            for k in 1:obs_dim
                acc += C[k, i] * λ[k] * C[k, j]
            end
            btd.H_diag[t][i, j] -= acc
        end
    end

    return nothing
end

"""
    _compute_gradient_poisson!(grad, ws, lds, y, x)

Compute the gradient of the log-likelihood for Poisson LDS.
Fills `grad` (latent_dim × tsteps) matrix with gradient values.
"""
function _compute_gradient_poisson!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    A = lds.state_model.A
    b = lds.state_model.b
    C = lds.obs_model.C
    d = lds.obs_model.d
    x0 = lds.state_model.x0

    tsteps = size(y, 2)
    latent_dim, obs_dim = lds.latent_dim, lds.obs_dim

    # Cholesky factors from cached PDMats (upper triangular factor)
    Q_chol_U = ws.Q_PD[].chol.U
    P0_chol_U = ws.P0_PD[].chol.U

    # Reuse workspace temp vectors
    Cx_t = ws.dyt           # obs_dim
    exp_term = ws.temp_dy   # obs_dim
    state_diff = ws.dxt     # latent_dim
    temp_grad = ws.tmp1     # latent_dim
    common_term = ws.tmp2   # latent_dim

    @views for t in 1:tsteps
        # Observation term: C' * (y - exp(C*x + d))
        mul!(Cx_t, C, x[:, t])
        @. exp_term = exp(Cx_t + d)
        exp_term .= y[:, t] .- exp_term
        mul!(common_term, C', exp_term)

        if t == 1
            # First timestep: common_term + A' * Q⁻¹ * (x₂ - A*x₁ - b) - P0⁻¹ * (x₁ - x0)
            mul!(state_diff, A, x[:, t])
            state_diff .= x[:, 2] .- state_diff .- b

            # Q⁻¹ * state_diff via Cholesky: solve Q_chol_U' * Q_chol_U * z = state_diff
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)

            mul!(grad[:, t], A', temp_grad)
            grad[:, t] .+= common_term

            # Subtract P0⁻¹ * (x₁ - x0)
            state_diff .= x[:, t] .- x0
            copyto!(temp_grad, state_diff)
            ldiv!(P0_chol_U', temp_grad)
            ldiv!(P0_chol_U, temp_grad)
            grad[:, t] .-= temp_grad

        elseif t == tsteps
            # Last timestep: common_term - Q⁻¹ * (xₜ - A*xₜ₋₁ - b)
            mul!(state_diff, A, x[:, t - 1])
            state_diff .= x[:, t] .- state_diff .- b

            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)

            grad[:, t] .= common_term .- temp_grad
        else
            # Middle timesteps
            # common_term + A' * Q⁻¹ * (xₜ₊₁ - A*xₜ - b) - Q⁻¹ * (xₜ - A*xₜ₋₁ - b)

            # Forward term: A' * Q⁻¹ * (xₜ₊₁ - A*xₜ - b)
            mul!(state_diff, A, x[:, t])
            state_diff .= x[:, t + 1] .- state_diff .- b
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)
            mul!(grad[:, t], A', temp_grad)
            grad[:, t] .+= common_term

            # Backward term: -Q⁻¹ * (xₜ - A*xₜ₋₁ - b)
            mul!(state_diff, A, x[:, t - 1])
            state_diff .= x[:, t] .- state_diff .- b
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)
            grad[:, t] .-= temp_grad
        end
    end

    return grad
end

"""
    smooth!(lds, fs, y, sws; max_iter=20, tol=1e-6)

Poisson LDS smoothing using iterative Newton with block tridiagonal solve.
Uses `SmoothWorkspace` for pre-allocated buffers.

Since the Poisson log-likelihood is non-quadratic, multiple Newton iterations
are required (unlike Gaussian LDS which converges in one step).
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    n_active = D * tsteps
    btd = sws.btd

    compute_smooth_constants!(sws, lds)

    x = fs.x_smooth

    if all(fs.E_z .== 0)
        @views x[:, 1] .= lds.state_model.x0
        @views for t in 2:tsteps
            mul!(x[:, t], lds.state_model.A, x[:, t - 1])
            x[:, t] .+= lds.state_model.b
        end
    else
        copyto!(x, fs.E_z)
    end

    # Active-length views into (possibly) oversized workspace buffers.
    X0 = view(sws.X₀, 1:n_active)
    grad_active = view(sws.grad_buf, :, 1:tsteps)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    ls = BackTrackingLS{T}()
    g = grad_active
    p = reshape(X0, D, tsteps)

    ϕ!() = _loglikelihood_ws(x, lds, y, sws)

    compute_grad! = (gcur, xcur) -> begin
        _compute_gradient_poisson!(gcur, sws, lds, y, xcur)
        return nothing
    end

    build_hess! = (xcur) -> begin
        _fill_hessian_blocks_poisson!(sws, lds, xcur)
        _negate_blocks!(btd, tsteps)
        return nothing
    end

    solve_dir! =
        (pcur, gcur) -> begin
            gvec = vec(gcur)
            pvec = vec(pcur)
            copyto!(pvec, gvec)
            # Negated Hessian is SPD at the MAP — use the SPD-specialised
            # solve so small `latent_dim` (≤ 8) routes to LAPACK `pbsv`.
            block_tridiagonal_solve_spd!(
                pvec, neg_sub_v, neg_diag_v, neg_super_v, gvec, btd
            )
            return nothing
        end

    newton_smooth!(
        Val(:max),
        x,
        g,
        p,
        compute_grad!,
        build_hess!,
        solve_dir!,
        ϕ!,
        ls;
        max_iter=max_iter,
        tol=tol,
    )

    _fill_hessian_blocks_poisson!(sws, lds, x)
    _negate_blocks!(btd, tsteps)

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    # `block_tridiagonal_inverse_logdet!` blocks are symmetric in exact
    # arithmetic but carry ~1e-12 asymmetry from the forward/back sweeps;
    # matches `gaussian.jl:780`. Without this, the aggregator's
    # `PDMat(copy(S0_sum))` can trip `ishermitian` downstream.
    @views for i in 1:tsteps
        Symmetrize!(fs.p_smooth[:, :, i])
    end

    return fs
end

"""
    smooth!(lds, tfs, y, sws_pool; max_iter=20, tol=1e-6)

Multi-trial Poisson LDS smoothing. Each task in `sws_pool` owns one workspace; trials
are partitioned across tasks via `@spawn`/`fetch`.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    ntrials = length(y)

    if ntrials == 1
        smooth!(lds, tfs[1], y[1], sws_pool[1]; max_iter=max_iter, tol=tol)
        return tfs
    end

    ntasks = min(ntrials, length(sws_pool))
    chunksize = cld(ntrials, ntasks)

    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue
        @spawn begin
            sws = sws_pool[i]
            for trial in lo:hi
                smooth!(lds, tfs[trial], y[trial], sws; max_iter=max_iter, tol=tol)
            end
        end
    end

    return tfs
end

"""
    smooth!(lds, tfs, y; max_iter=20, tol=1e-6)

Convenience method that creates a workspace pool sized at `max(T_i)`.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for _ in 1:npool]
    return smooth!(lds, tfs, y, sws_pool; max_iter=max_iter, tol=tol)
end

"""
    estep!(lds, suf, tfs, y, u_seq, v_seq, sws_pool; max_iter=20, tol=1e-6)

Suf-based Poisson E-step. Smooths, aggregates state-side sufficient
statistics into `suf` from each trial's smoother output (`x_smooth`,
`p_smooth`, `p_smooth_tt1`), and returns the suf-based ELBO. The legacy
per-trial `sufficient_statistics!(tfs)` call is skipped — the suf-based
M-step doesn't need `fs.E_z` / `fs.E_zz` / `fs.E_zz_prev`, and the
Poisson emission update now reads `fs.x_smooth` directly.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    smooth!(lds, tfs, y, sws_pool; max_iter=max_iter, tol=tol)
    _aggregate_td_suff_stats!(suf, tfs, lds, u_seq, v_seq, y, sws_pool[1])
    elbo = calculate_elbo(lds, suf, tfs, y, sws_pool)
    return elbo
end

"""
    fit!(plds, y; max_iter=100, tol=1e-6, progress=true, newton_max_iter=20, newton_tol=1e-6)

Fit a Poisson LDS via Laplace-EM.

# Arguments
- `plds`: the Poisson LDS model (modified in place)
- `y`: observations. Two shapes accepted:
    * `AbstractMatrix{T}` of size `(obs_dim, T)` — single trial
    * `AbstractVector{<:AbstractMatrix{T}}` — multi-trial, each `(obs_dim, T_i)`,
      ragged trial lengths allowed

# Keywords
- `max_iter`: maximum EM iterations
- `tol`: convergence tolerance on ELBO change
- `progress`: show progress bar
- `newton_max_iter`: Newton iterations per E-step inner solve
- `newton_tol`: Newton convergence tolerance
"""
function fit!(
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    obs_dim = plds.obs_dim
    latent_dim = plds.latent_dim
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)

    tfs = initialize_FilterSmooth(plds, tsteps_per_trial)::TrialFilterSmooth{T}

    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, latent_dim, obs_dim, T_max) for _ in 1:npool]

    # Suf-based state-side M-step (mirrors the Gaussian TD fit path). Poisson
    # has no controls, so `u_seq` / `v_seq` are zero-row matrices. The const
    # blocks (bias-row entries, obs_yy_const, …) are precomputed once; the
    # `obs_*` blocks are written by the aggregator but unread by the Poisson
    # M-step (emission stays LBFGS), which is a tiny constant overhead.
    suf = _initialize_td_sufficient_statistics(T, plds, tsteps_per_trial)
    u_seq = [zeros(T, 0, Ti) for Ti in tsteps_per_trial]
    v_seq = [zeros(T, 0, Ti) for Ti in tsteps_per_trial]
    _td_init_const_blocks!(sws_pool[1], plds, tsteps_per_trial, y, u_seq, v_seq)

    elbos = Vector{T}(undef, max_iter)

    prog = if progress
        Progress(
            max_iter;
            desc="Fitting Poisson LDS via LaPlaceEM...",
            barlen=50,
            showspeed=true,
        )
    else
        nothing
    end

    for iter in 1:max_iter
        elbos[iter] = estep!(
            plds,
            suf,
            tfs,
            y,
            u_seq,
            v_seq,
            sws_pool;
            max_iter=newton_max_iter,
            tol=T(newton_tol),
        )
        mstep!(plds, suf, tfs, y, sws_pool)

        prog !== nothing && next!(prog)

        if iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol
            resize!(elbos, iter)
            break
        end
    end

    prog !== nothing && finish!(prog)
    return elbos
end

function fit!(
    plds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}; kwargs...
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    return fit!(plds, [y]; kwargs...)
end
