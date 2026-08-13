#=============================================================================
Poisson Observations

    Emission kernels: observation_loglikelihood!(cc, z, λ, lds, x, y, t[, uy])
                      observation_gradient!(out, cc, buf, lds, x, y, t[, uy])
                      observation_hessian!(out, cc, z, λ, lds, x, y, t[, α])

    E-Step: Q_obs!(sws, lds, suf)

    M-Step: update_observation_model!(plds, tfs, y, sws_pool, w)
=============================================================================#

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
    sws::SmoothWorkspace{T},                      # provides the Poisson Q_obs scratch
    plds::LinearDynamicalSystem{T,S,O},            # for obs_model.C, .d and .D
    E_z::AbstractMatrix{T},                       # state_dim × T
    p_smooth::AbstractArray{T,3},                 # state_dim × state_dim × T
    y::AbstractMatrix{T},                         # obs_dim × T
    uy::Union{Nothing,AbstractMatrix}=nothing;    # obs inputs (uy_dim × T) or nothing
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    d = plds.obs_model.d
    C = plds.obs_model.C
    D = plds.obs_model.D

    obs_dim, _ = size(C)
    tsteps = size(y, 2)

    # workspace buffers
    h = sws.elbo.h_obs::Vector{T}            # obs_dim
    rho = sws.elbo.rho_obs::Vector{T}            # obs_dim
    CP = sws.elbo.CP_obs::Matrix{T}            # obs_dim × state_dim
    CEz = sws.elbo.CEz_obs::Vector{T}            # obs_dim

    Q_val = zero(T)

    @views for t in 1:tsteps
        wt = isnothing(weights) ? one(T) : weights[t]

        Ez_t = E_z[:, t]                 # state_dim
        P_t = p_smooth[:, :, t]         # state_dim × state_dim
        y_t = y[:, t]                   # obs_dim

        # CEz = C * Ez_t
        mul!(CEz, C, Ez_t)

        # h = CEz + d (+ D v_t)
        @. h = CEz + d
        if uy !== nothing
            mul!(h, D, uy[:, t], one(T), one(T))
        end

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
    update_observation_model!(plds, tfs, y, sws_pool, w; uy=nothing)

Update the observation model parameters `[C d D]` of a PLDS model via LBFGS.
`uy` is the per-trial vector of observation-input matrices (or `nothing`).
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing;
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    plds.fit_bool[5] || return nothing

    sws = sws_pool[1]       # f(params) is sequential; one workspace suffices
    obs_dim = plds.obs_dim
    latent_dim = plds.latent_dim
    uy_dim = plds.uy_dim
    Dp1 = latent_dim + 1
    reg_dim = Dp1 + uy_dim          # columns of the stacked W = [C d D]
    n_W = obs_dim * reg_dim         # total params; identical to vec([C d D]) length

    # Param vector layout: vcat(vec(C), d, vec(D)) == vec([C d D]) column-major.
    params = vcat(vec(plds.obs_model.C), plds.obs_model.d, vec(plds.obs_model.D))

    # MN-only prior on the stacked emission matrix W = [C d D] (Poisson has no IW
    # counterpart since there is no observation-noise covariance):
    CD_prior = plds.obs_model.CD_prior

    function f(params::Vector{T})
        W_view = reshape(view(params, 1:n_W), obs_dim, reg_dim)
        @views C_view = W_view[:, 1:latent_dim]
        @views d_view = W_view[:, Dp1]
        @views D_view = W_view[:, (Dp1 + 1):reg_dim]

        copyto!(plds.obs_model.C, C_view)
        copyto!(plds.obs_model.d, d_view)
        copyto!(plds.obs_model.D, D_view)

        acc = zero(T)
        ntrials = length(tfs)

        for trial in 1:ntrials
            fs = tfs[trial]
            weights = isnothing(w) ? nothing : w[trial]
            uy_trial = isnothing(uy) ? nothing : uy[trial]
            # `x_smooth` is the same value as `E_z` for a Gaussian state model
            # — see `gradient_observation_model!` for context.
            acc += Q_obs!(
                sws, plds, fs.x_smooth, fs.p_smooth, y[trial], uy_trial; weights=weights
            )
        end

        f_prior = zero(T)
        if CD_prior !== nothing
            Wm = W_view .- CD_prior.M₀
            f_prior = T(0.5) * sum(Wm .* (Wm * CD_prior.Λ))
        end

        return -acc + f_prior
    end

    function g!(grad::Vector{T}, params::Vector{T})
        W_view = reshape(view(params, 1:n_W), obs_dim, reg_dim)
        @views C_view = W_view[:, 1:latent_dim]
        @views d_view = W_view[:, Dp1]
        @views D_view = W_view[:, (Dp1 + 1):reg_dim]
        gradient_observation_model!(grad, C_view, d_view, D_view, tfs, y, uy, sws_pool, w)
        if CD_prior !== nothing
            grad_W_view = reshape(view(grad, 1:n_W), obs_dim, reg_dim)
            grad_W_view .+= (W_view .- CD_prior.M₀) * CD_prior.Λ
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
    result_W = reshape(result.minimizer[1:n_W], obs_dim, reg_dim)
    @views plds.obs_model.C .= result_W[:, 1:latent_dim]
    @views plds.obs_model.d .= result_W[:, Dp1]
    @views plds.obs_model.D .= result_W[:, (Dp1 + 1):reg_dim]

    return nothing
end

"""
    observation_loglikelihood!(cc, z, λ, lds, x, y, t[, uy])

Poisson emission term: with rate `λ = exp(Cx_t + d + D v_t)`,
`log p(y_t|x_t) = y⋅log(λ) - sum(λ) - sum(log(y!))`. `z` and `λ` are `obs_dim`
scratch vectors for the linear predictor and the rate. `uy` (optional) supplies
the observation input `v_t`; `nothing` or a zero-row matrix skips the `D v_t`
term. The cache argument is unused (no covariance term).
"""
function observation_loglikelihood!(
    ::SmoothConstants{T},
    z::AbstractVector{T},
    λ::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:PoissonObservationModel{T0}}
    C = lds.obs_model.C
    d = lds.obs_model.d

    # z = Cx + d (+ D v) ; λ = exp(z)
    @views mul!(z, C, x[:, t])
    if uy !== nothing
        @views mul!(z, lds.obs_model.D, uy[:, t], one(T), one(T))
    end
    z .+= d
    @. λ = exp(z)

    # y⋅z - λ - log(y!)  (loggamma(n+1) = log(n!) for real n≥0)
    yt = view(y, :, t)
    return dot(yt, z) - sum(λ) - sum(yi -> loggamma(yi + one(T)), yt)
end

"""
    observation_gradient!(out, cc, buf, lds, x, y, t[, uy])

Poisson emission gradient w.r.t. the latent `x_t`: `out = C'(y_t - λ_t)` with
`λ_t = exp(Cx_t + d + D v_t)`. The `D v_t` term is constant in `x_t`, so it
enters only through the rate `λ`. The cache argument is unused (no covariance
term); `uy` (optional) supplies `v_t`.
"""
function observation_gradient!(
    out::AbstractVector{T},
    ::SmoothConstants{T},
    buf::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:PoissonObservationModel{T0}}
    C = lds.obs_model.C
    d = lds.obs_model.d
    @views mul!(buf, C, x[:, t])
    if uy !== nothing
        @views mul!(buf, lds.obs_model.D, uy[:, t], one(T), one(T))
    end
    @views buf .= y[:, t] .- exp.(buf .+ d)
    return mul!(out, C', buf)
end

"""
    observation_hessian!(out, cc, z, λ, lds, x, y, t[, α, uy])

Poisson emission curvature: `out .+= α .* (-C' diag(λ_t) C)` with
`λ_t = exp(C x_t + d + D v_t)` — independent of `y` for the canonical log link.
The `D v_t` term enters only through the rate `λ`. `z` and `λ` are `obs_dim`
scratch for the linear predictor and the rate; `cc` is unused (no covariance in
the emission term); `uy` (optional) supplies `v_t`.
"""
function observation_hessian!(
    out::AbstractMatrix{T},
    ::SmoothConstants{T},
    z::AbstractVector{T},
    λ::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    α::T=one(T),
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:PoissonObservationModel{T0}}
    C = lds.obs_model.C
    d = lds.obs_model.d
    obs_dim, latent_dim = size(C)

    @views mul!(z, C, x[:, t])
    if uy !== nothing
        @views mul!(z, lds.obs_model.D, uy[:, t], one(T), one(T))
    end
    @. λ = exp(z + d)

    # out .+= α * (-C' diag(λ) C), allocation-free (O(latent² · obs) per call).
    for j in 1:latent_dim, i in 1:latent_dim
        acc = zero(T)
        for k in 1:obs_dim
            acc += C[k, i] * λ[k] * C[k, j]
        end
        out[i, j] -= α * acc
    end
    return out
end
