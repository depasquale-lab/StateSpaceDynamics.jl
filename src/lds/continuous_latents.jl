#=============================================================================
Continuous (Linear Gaussian) latents

    Log-Likelihood kernels: state_loglikelihood!(cc, dxt, tmp, lds, x, t[, ux])
                            observation_loglikelihood!(cc, b1, b2, lds, x, y, t[, uy])
                            joint_loglikelihood!(ll, ws, cc, lds, x, y[, ux, uy])

    Gradient kernels:       observation_gradient!(out, cc, buf, lds, x, y, t[, uy])
                            gradient!(grad, ws, lds, x, y[, ux, uy])

    Hessian kernels:        observation_hessian!(out, cc, buf1, buf2, lds, x, y, t[, α])
                            hessian!(sws, lds, x, y)

    E-Step: Q_state!(sws, lds, suf)

    M-Step: update_initial_state_mean!(lds, suf)
            update_initial_state_covariance!(lds, suf, sws)
            update_A_b!(lds, suf, sws)
            update_Q!(lds, suf, sws)
=============================================================================#

# In-place whitening solve `v := U⁻ᵀ v` against a Cholesky Σ = U'U, so that
# `sum(abs2, v)` afterwards is the quadratic form `v'Σ⁻¹v`. 
@inline function _whiten!(
    chol::Cholesky{T,Matrix{T}}, v::StridedVector{T}
) where {T<:LinearAlgebra.BlasFloat}
    if chol.uplo === 'U'
        LinearAlgebra.LAPACK.trtrs!('U', 'T', 'N', chol.factors, v)
    else
        LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', chol.factors, v)
    end
    return v
end
_whiten!(chol::Cholesky, v::AbstractVector) = ldiv!(chol.U', v)

"""
    _transition_residual!(out, lds, x, t[, ux])

Write the dynamics residual `x_t - A x_{t-1} - b - B u_{t-1}` into `out`
(length `latent_dim`). Pass `ux` (state inputs, `t`-indexed like `x`) to
include the `B u_{t-1}` term; `nothing` (default) skips it. Requires `t ≥ 2`.
"""
@inline function _transition_residual!(
    out::AbstractVector{T},
    lds::LinearDynamicalSystem,
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real}
    @views mul!(out, lds.state_model.A, x[:, t - 1])
    if ux !== nothing
        @views mul!(out, lds.state_model.B, ux[:, t - 1], one(T), one(T))
    end
    @views out .= x[:, t] .- out .- lds.state_model.b
    return out
end

"""
    state_loglikelihood!(cc, dxt, tmp, lds, x, t[, ux])

State-model (prior/transition) contribution to the complete-data log-likelihood
at timestep `t`:

- `t == 1`: `cP0 - 0.5‖P0^{-1/2}(x_1 - x_0)‖²`
- `t ≥ 2`:  `cQ - 0.5‖Q^{-1/2}(x_t - A x_{t-1} - b - B u_{t-1})‖²`

`cc` is a [`SmoothConstants`](@ref) holding the Cholesky factors and
normalizers; `dxt` and `tmp` are `latent_dim` scratch vectors (overwritten).
Pass `ux` (state inputs, `t`-indexed like `x`) to include the `B u_{t-1}`
term; `nothing` (default) or a zero-row matrix skips it.
"""
function state_loglikelihood!(
    cc::SmoothConstants{T},
    dxt::AbstractVector{T},
    tmp::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:AbstractObservationModel{T0}}
    if t == 1
        @views dxt .= x[:, 1] .- lds.state_model.x0
        _whiten!(cc.P0_PD.chol, dxt)
        return cc.cP0 - T(0.5) * sum(abs2, dxt)
    else
        _transition_residual!(tmp, lds, x, t, ux)
        _whiten!(cc.Q_PD.chol, tmp)
        return cc.cQ - T(0.5) * sum(abs2, tmp)
    end
end

"""
    observation_loglikelihood!(cc, buf1, buf2, lds, x, y, t[, uy])

Emission-model contribution `log p(y_t | x_t)` to the complete-data
log-likelihood at timestep `t`. Dispatches on the observation model type `O`
(via `lds::LinearDynamicalSystem{T,S,O}`); a custom observation model plugs
into `joint_loglikelihood!` by adding a method here.

- `cc`: a [`SmoothConstants`](@ref) with Cholesky factors / normalizers
  (unused by models whose emission term needs no covariance, e.g. Poisson).
- `buf1`, `buf2`: two `obs_dim` scratch vectors (overwritten); each model uses
  what it needs (Gaussian: residual; Poisson: linear predictor + rate).
- `uy` (optional): observation inputs, `t`-indexed like `y`; `nothing` or a
  zero-row matrix skips the `D u_t` term.
"""
function observation_loglikelihood! end

"""
    joint_loglikelihood!(ll, ws, cc, lds, x, y[, ux, uy])

Per-timestep complete-data log-likelihood of one LDS, written into `ll`:
`ll[t] = log p(y_t | x_t) + log p(x_t | x_{t-1})` (or `+ log p(x_1)` at
`t == 1`). Generic over the observation model — the emission term comes from
`observation_loglikelihood!` — and over the workspace: the single-LDS path
passes a `SmoothWorkspace`, the SLDS path a `SLDSSmoothWorkspace` (with `cc`
the component's `SmoothConstants`), both consuming the shared `NewtonBuffers`
scratch in `ws.opt`. `ux` / `uy` are optional control inputs (`nothing` or
zero-row matrices skip the `B u` / `D uy` terms).

Notes:
- Normalization terms (logdet + log(2π)) are included. These are constant w.r.t.
  `x`, but **not** constant across SLDS discrete states when `Q`/`R` differ by
  state.
"""
function joint_loglikelihood!(
    ll::AbstractVector{T},
    ws::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    cc::SmoothConstants{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:AbstractObservationModel{T0}}
    tsteps = size(y, 2)
    @assert length(ll) == tsteps

    opt = ws.opt

    for t in 1:tsteps
        ll_t = observation_loglikelihood!(
            cc, opt.temp_dy, opt.temp_solve_R, lds, x, y, t, uy
        )
        ll_t += state_loglikelihood!(cc, opt.temp_dx, opt.temp_solve_Q, lds, x, t, ux)
        ll[t] = ll_t
    end

    return ll
end

"""
    observation_gradient!(out, cc, buf, lds, x, y, t[, uy])

Emission-model contribution `∂ log p(y_t | x_t) / ∂x_t` written into `out`
(length `latent_dim`). Dispatches on the observation model type `O` (via
`lds::LinearDynamicalSystem{T,S,O}`); a custom observation model plugs into
`gradient!` (both the single-LDS and the SLDS weighted form) by adding a
method here.

- `cc`: a [`SmoothConstants`](@ref) with Cholesky-derived terms (Gaussian
  uses the cached `C_inv_R = C'R⁻¹`; models without a covariance ignore it).
- `buf`: one `obs_dim` scratch vector (overwritten).
- `uy` (optional): observation inputs, `t`-indexed like `y`; `nothing` or a
  zero-row matrix skips the `D u_t` term.
"""
function observation_gradient! end

"""
    gradient!(grad, ws, lds, x, y[, ux, uy])
    gradient!(ws, lds, x, y[, ux, uy])

Gradient of the complete-data log-likelihood with respect to the latent path
`x`, written into `grad` (`latent_dim × tsteps`; the convenience form uses the
active view of `ws.opt.grad_buf` and returns it). Generic over the observation
model — the emission term comes from `observation_gradient!`, while the state
side (prior / incoming / outgoing transition factors) is shared:

- `grad[:, t] = obs_grad(t) + A'Q⁻¹ r_{t+1} - Q⁻¹ r_t`   (middle steps)
- at `t = 1` the `-Q⁻¹ r_t` term is replaced by `-P0⁻¹(x_1 - x_0)`
- at `t = T` the `A'Q⁻¹ r_{t+1}` term is absent

with `r_t = x_t - A x_{t-1} - b - B u_{t-1}`. Uses the Cholesky-derived
templates cached by `compute_smooth_constants!`; requires `tsteps ≥ 2`.
"""
function gradient!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps = size(x, 2)

    cc = ws.consts
    A_inv_Q = cc.A_inv_Q          # A'Q⁻¹
    neg_P0_inv = cc.x_t           # -P0⁻¹
    neg_Q_inv = cc.xt_given_xt_1  # -Q⁻¹

    dxt = ws.opt.dxt
    dxt_next = ws.opt.dxt_next
    obs_buf = ws.opt.dyt
    tmp1 = ws.opt.tmp1
    tmp2 = ws.opt.tmp2
    tmp3 = ws.opt.tmp3

    # First time step: emission + prior + outgoing factor at t = 2
    observation_gradient!(tmp1, cc, obs_buf, lds, x, y, 1, uy)
    @views dxt .= x[:, 1] .- lds.state_model.x0
    mul!(tmp3, neg_P0_inv, dxt)
    _transition_residual!(dxt_next, lds, x, 2, ux)
    mul!(tmp2, A_inv_Q, dxt_next)
    @views grad[:, 1] .= tmp1 .+ tmp2 .+ tmp3

    # Middle steps: emission + incoming factor at t + outgoing factor at t + 1
    @views for t in 2:(tsteps - 1)
        observation_gradient!(tmp1, cc, obs_buf, lds, x, y, t, uy)
        _transition_residual!(dxt, lds, x, t, ux)
        mul!(tmp3, neg_Q_inv, dxt)
        _transition_residual!(dxt_next, lds, x, t + 1, ux)
        mul!(tmp2, A_inv_Q, dxt_next)
        grad[:, t] .= tmp1 .+ tmp3 .+ tmp2
    end

    # Last time step: emission + incoming factor at t = T
    observation_gradient!(tmp1, cc, obs_buf, lds, x, y, tsteps, uy)
    _transition_residual!(dxt, lds, x, tsteps, ux)
    mul!(tmp3, neg_Q_inv, dxt)
    @views grad[:, tsteps] .= tmp1 .+ tmp3

    return grad
end

function gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    grad = view(ws.opt.grad_buf, :, 1:size(x, 2))
    return gradient!(grad, ws, lds, x, y, ux, uy)
end

"""
    observation_hessian!(out, cc, buf1, buf2, lds, x, y, t[, α])

Emission-model contribution `∂² log p(y_t | x_t) / ∂x_t²` **accumulated** into
`out` (`latent_dim × latent_dim`) with weight `α`: `out .+= α .* hess_t`. The
add-with-weight semantics let the same kernel serve both the single-LDS
`hessian!` (α = 1, `out` pre-filled with the state-side block) and the SLDS
`hessian!` (α = w[k,t], accumulating across mixture components).

Dispatches on the observation model type `O` (via
`lds::LinearDynamicalSystem{T,S,O}`) — the curvature companion to
`observation_gradient!`: a custom observation model plugs into both `hessian!`
forms by adding a method here, without touching the shared state-side Hessian
blocks.

Uniform interface:
- `cc`: a [`SmoothConstants`](@ref) with Cholesky-derived templates
  (Gaussian uses the cached `yt_given_xt = -C'R⁻¹C`; Poisson ignores it).
- `buf1`, `buf2`: two `obs_dim` scratch vectors (overwritten); each model uses
  what it needs (Gaussian: none; Poisson: linear predictor + rate).
- `y` is unused by the built-in models (Gaussian/Poisson-canonical curvature is
  observation-independent) but part of the interface for models whose curvature
  depends on `y`.

See the `GaussianObservationModel` / `PoissonObservationModel` methods in
`gaussian_observations.jl` / `poisson_observations.jl` for the pattern to
follow.
"""
function observation_hessian! end

"""
    _state_hessian_blocks!(btd, cc, tsteps)

Write the state-side (prior/transition) Hessian blocks — identical for every
observation model — into `btd.H_diag` / `H_sub` / `H_super`:

- `H_sub[i] = Q⁻¹A`, `H_super[i] = (Q⁻¹A)'` for all i
- `H_diag[1] = -A'Q⁻¹A - P0⁻¹`
- `H_diag[t] = -A'Q⁻¹A - Q⁻¹` (middle), `H_diag[T] = -Q⁻¹`

Uses the templates cached on `cc` by `compute_smooth_constants!`.
Overwrites the diagonal blocks — callers add the emission curvature
afterwards via `observation_hessian!`. Requires `tsteps ≥ 2` (matching the
Newton smoother's contract).
"""
function _state_hessian_blocks!(btd, cc::SmoothConstants{T}, tsteps::Int) where {T<:Real}
    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], cc.H_sub_entry)
        copyto!(btd.H_super[i], cc.H_super_entry)
    end

    btd.H_diag[1] .= cc.xt1_given_xt .+ cc.x_t
    for t in 2:(tsteps - 1)
        btd.H_diag[t] .= cc.xt1_given_xt .+ cc.xt_given_xt_1
    end
    btd.H_diag[tsteps] .= cc.xt_given_xt_1

    return nothing
end

"""
    hessian!(sws, lds, x, y)

Fill `sws.btd.H_diag`, `H_sub`, `H_super` with the complete-data log-likelihood
Hessian blocks w.r.t. the latent path (length derived from `size(y, 2)`).
Returns nothing — the sparse form is **not** built here because the Newton
solver consumes blocks directly. Workspace buffers may be sized for a longer
trial; only the first `tsteps` blocks are written, which keeps this hot path
safe for ragged-length fitting.

Generic over the observation model — the state-side blocks come from
`_state_hessian_blocks!` and the emission curvature from
`observation_hessian!`, so supporting a new observation model here only
requires a new `observation_hessian!` method, not a new `hessian!` method.
Requires `compute_smooth_constants!(sws, lds)` to have been called.
"""
function hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps = size(y, 2)
    btd = sws.btd
    cc = sws.consts

    _state_hessian_blocks!(btd, cc, tsteps)
    for t in 1:tsteps
        observation_hessian!(
            btd.H_diag[t], cc, sws.elbo.rho_obs, sws.elbo.h_obs, lds, x, y, t
        )
    end

    return nothing
end

"""
    Q_state!(ws, lds, E_z, E_zz, E_zz_prev, ux)

State Q-term for an LDS with affine dynamics `x_t ~ N(A x_{t-1} + b + B u_{t-1}, Q)`.
In-place version of `Q_state` that uses pre-allocated buffers from `SmoothWorkspace`.
Uses cached Cholesky factors from `compute_smooth_constants!`. `ux` is the per-trial
latent-input matrix `(ux_dim, T_i)`; pass a `0×T_i` matrix when no inputs.
"""
function Q_state!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    ux::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tstep = size(E_z, 2)
    D = lds.latent_dim
    ux_dim = size(ux, 1)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0

    # Use cached Cholesky factors (already computed by compute_smooth_constants!)
    Q_U = ws.consts.Q_PD.chol.U
    P0_U = ws.consts.P0_PD.chol.U

    log_det_Q = logdet(ws.consts.Q_PD)
    log_det_P0 = logdet(ws.consts.P0_PD)

    temp = ws.elbo.temp
    sum_E_zz = ws.elbo.sum_E_zz
    sum_E_zzm1 = ws.elbo.sum_E_zzm1
    sum_E_cross = ws.elbo.sum_E_cross
    sum_mu_t = ws.elbo.sum_mu_t
    sum_mu_tm1 = ws.elbo.sum_mu_tm1
    temp2 = ws.elbo.temp2

    # Initial-state part: temp = E_zz[:,:,1] - E_z[:,1]*x0' - x0*E_z[:,1]' + x0*x0'
    fill!(temp, zero(T))
    @views begin
        temp .+= E_zz[:, :, 1]
        mul!(temp, E_z[:, 1:1], x0', -one(T), one(T))
        mul!(temp, x0, E_z[:, 1:1]', -one(T), one(T))
        mul!(temp, x0, x0', one(T), one(T))
    end
    ldiv!(P0_U', temp)
    ldiv!(P0_U, temp)
    Q_val = T(-0.5) * (log_det_P0 + tr(temp))

    # Transition part: accumulate sums over t=2:tstep
    fill!(sum_E_zz, zero(T))
    fill!(sum_E_zzm1, zero(T))
    fill!(sum_E_cross, zero(T))
    fill!(sum_mu_t, zero(T))
    fill!(sum_mu_tm1, zero(T))

    #=
    Input-specific accumulators (only allocated when ux_dim > 0). Allocating
    0-element arrays here would still cost an `Array` struct each call,
    which adds up to thousands of trivial allocations across a fit.
    =#
    has_input = ux_dim > 0
    sum_u = has_input ? zeros(T, ux_dim) : Vector{T}()
    sum_mu_t_u = has_input ? zeros(T, D, ux_dim) : Matrix{T}(undef, 0, 0)
    sum_mu_tm1_u = has_input ? zeros(T, D, ux_dim) : Matrix{T}(undef, 0, 0)
    sum_uu = has_input ? zeros(T, ux_dim, ux_dim) : Matrix{T}(undef, 0, 0)

    @views for t in 2:tstep
        sum_E_zz .+= E_zz[:, :, t]
        sum_E_zzm1 .+= E_zz[:, :, t - 1]
        sum_E_cross .+= E_zz_prev[:, :, t]
        sum_mu_t .+= E_z[:, t]
        sum_mu_tm1 .+= E_z[:, t - 1]

        if has_input
            ux_tm1 = ux[:, t - 1]
            sum_u .+= ux_tm1
            BLAS.ger!(one(T), E_z[:, t], ux_tm1, sum_mu_t_u)
            BLAS.ger!(one(T), E_z[:, t - 1], ux_tm1, sum_mu_tm1_u)
            BLAS.ger!(one(T), ux_tm1, ux_tm1, sum_uu)
        end
    end

    # No-input batched terms:
    #   temp = sum_E_zz - A·sum_E_cross' - sum_E_cross·A' + A·sum_E_zzm1·A'
    copyto!(temp, sum_E_zz)
    mul!(temp, A, sum_E_cross', -one(T), one(T))
    mul!(temp, sum_E_cross, A', -one(T), one(T))
    mul!(temp2, A, sum_E_zzm1)
    mul!(temp, temp2, A', one(T), one(T))

    # Bias terms (b alone):
    mul!(temp, sum_mu_t, b', -one(T), one(T))
    mul!(temp, b, sum_mu_t', -one(T), one(T))
    mul!(ws.opt.tmp1, A, sum_mu_tm1)
    mul!(temp, ws.opt.tmp1, b', one(T), one(T))
    mul!(temp, b, ws.opt.tmp1', one(T), one(T))
    mul!(temp, b, b', T(tstep - 1), one(T))

    #=
    Input cross terms (`Bu_{t-1} := B u_{t-1}`). All terms here are
    contributions to `Σ_t E[(x_t - A x_{t-1} - b - B u_{t-1})(...)']` that
    involve at least one `B u_{t-1}` factor.
    =#
    if ux_dim > 0
        # -= sum_mu_t_u · B'  and  -= B · sum_mu_t_u'
        mul!(temp, sum_mu_t_u, B', -one(T), one(T))
        mul!(temp, B, sum_mu_t_u', -one(T), one(T))
        # += (A · sum_mu_tm1_u) · B'  and  += B · (A · sum_mu_tm1_u)'
        # Intermediate has shape (D × ux_dim); no fixed-size workspace buffer.
        A_sumXU = A * sum_mu_tm1_u
        mul!(temp, A_sumXU, B', one(T), one(T))
        mul!(temp, B, A_sumXU', one(T), one(T))
        # += b · (B · sum_u)'  and  += (B · sum_u) · b'
        B_sumu = B * sum_u  # D-vector
        mul!(temp, reshape(b, :, 1), reshape(B_sumu, 1, :), one(T), one(T))
        mul!(temp, reshape(B_sumu, :, 1), reshape(b, 1, :), one(T), one(T))
        # += B · sum_uu · B'
        B_sumuu = B * sum_uu  # D × ux_dim
        mul!(temp, B_sumuu, B', one(T), one(T))
    end

    # Solve Q \ temp
    ldiv!(Q_U', temp)
    ldiv!(Q_U, temp)
    Q_val += T(-0.5) * ((tstep - 1) * log_det_Q + tr(temp))

    return Q_val
end

# Backward-compatible no-input overload.
function Q_state!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    ux = zeros(T, 0, size(E_z, 2))
    return Q_state!(ws, lds, E_z, E_zz, E_zz_prev, ux)
end

"""
    Q_state!(sws, lds, suf)

Total log-likelihood Q-state term across all trials, computed from the
aggregated sufficient statistics in `suf`. Replaces the per-trial,
per-timestep loops of the legacy `Q_state!(sws, lds, E_z, E_zz, E_zz_prev, ux)`.

Identical value (up to floating-point) to summing the legacy form across
trials.
"""
function Q_state!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    ux_dim = lds.ux_dim
    dyn_reg_dim = D + 1 + ux_dim

    Q_U = sws.consts.Q_PD.chol.U
    P0_U = sws.consts.P0_PD.chol.U

    log_det_Q = logdet(sws.consts.Q_PD)
    log_det_P0 = logdet(sws.consts.P0_PD)

    N = suf.init_n
    dyn_n = suf.dyn_n

    log2π = log(T(2π))
    const_init = T(N) * D * log2π
    const_trans = T(dyn_n) * D * log2π

    # S_init = init_yy - μ x0' - x0 μ' + N x0 x0'    (μ = Σ x_init)
    S_init = sws.elbo.temp
    copyto!(S_init, suf.init_yy[].mat)
    μ_sum = vec(suf.init_xy)
    BLAS.ger!(-one(T), μ_sum, x0, S_init)
    BLAS.ger!(-one(T), x0, μ_sum, S_init)
    BLAS.ger!(T(N), x0, x0, S_init)

    ldiv!(P0_U', S_init)
    ldiv!(P0_U, S_init)
    Q_val = T(-0.5) * (const_init + T(N) * log_det_P0 + tr(S_init))

    # W = [A b B] (D × dyn_reg_dim)
    W = view(sws.reg.AB, :, 1:dyn_reg_dim)
    copyto!(view(W, :, 1:D), A)
    copyto!(view(W, :, D + 1), b)
    if ux_dim > 0
        copyto!(view(W, :, (D + 2):dyn_reg_dim), B)
    end

    S_trans = sws.elbo.temp                 # reuse (S_init no longer needed)
    copyto!(S_trans, suf.dyn_yy[].mat)
    mul!(S_trans, W, suf.dyn_xy, -one(T), one(T))
    mul!(S_trans, transpose(suf.dyn_xy), transpose(W), -one(T), one(T))
    # S_trans += W · dyn_xx · W'
    W_XX = view(sws.reg.Sxz, :, 1:dyn_reg_dim)
    mul!(W_XX, W, suf.dyn_xx[].mat)
    mul!(S_trans, W_XX, transpose(W), one(T), one(T))

    ldiv!(Q_U', S_trans)
    ldiv!(Q_U, S_trans)
    Q_val += T(-0.5) * (const_trans + T(dyn_n) * log_det_Q + tr(S_trans))

    return Q_val
end

function update_initial_state_mean!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[1] || return nothing
    inv_n = inv(T(suf.init_n))
    x0 = lds.state_model.x0
    for j in eachindex(x0)
        x0[j] = suf.init_xy[1, j] * inv_n
    end
    return nothing
end

function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[2] || return nothing
    D = lds.latent_dim
    x0 = lds.state_model.x0
    N = suf.init_n

    S0 = sws.reg.S0_sum                              # D × D scratch
    copyto!(S0, suf.init_yy[].mat)

    # Rank-1 updates inline (BLAS.ger! would need a contiguous μ vector and
    # `view(init_xy, 1, :)` allocates a SubArray header — small but nonzero).
    for j in 1:D
        μ_j = suf.init_xy[1, j]
        x0_j = x0[j]
        for i in 1:D
            μ_i = suf.init_xy[1, i]
            x0_i = x0[i]
            S0[i, j] += T(N) * x0_i * x0_j - x0_i * μ_j - μ_i * x0_j
        end
    end
    Symmetrize!(S0)

    if lds.state_model.P0_prior === nothing
        S0 ./= T(N)
    else
        Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
        # iw_map inlined: (Ψ + S0) / (ν + N + D + 1)
        denom = ν + T(N) + T(D + 1)
        for i in eachindex(S0)
            S0[i] = (Ψ[i] + S0[i]) / denom
        end
    end
    copyto!(lds.state_model.P0, S0)
    return nothing
end

function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[3] || return nothing
    D = lds.latent_dim
    ux_dim = lds.ux_dim
    AB_prior = lds.state_model.AB_prior

    if AB_prior === nothing
        #=
        Zero-alloc OLS fast path. `sws.reg.Sxz` is exactly (D × dyn_reg_dim);
        its transpose is the (dyn_reg_dim × D) view we ldiv! into. After
        the in-place solve, `sws.reg.Sxz` itself holds the transposed solution
        `transpose(dyn_xx \ dyn_xy)` = the W = [A b B] regression matrix.
        =#
        Sxz_T = transpose(sws.reg.Sxz)
        copyto!(Sxz_T, suf.dyn_xy)
        ldiv!(suf.dyn_xx[].chol, Sxz_T)
        W = sws.reg.Sxz
    else
        # MN-prior MAP path — keep `mn_map` (allocates) for now.
        W = mn_map(suf.dyn_xx[], suf.dyn_xy, AB_prior)
    end

    copyto!(lds.state_model.A, view(W, :, 1:D))
    copyto!(lds.state_model.b, view(W, :, D + 1))
    if ux_dim > 0
        copyto!(lds.state_model.B, view(W, :, (D + 2):(D + 1 + ux_dim)))
    end
    return nothing
end

function update_Q!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing
    D = lds.latent_dim
    ux_dim = lds.ux_dim

    # sws.reg.AB is exactly (D × dyn_reg_dim); no view needed.
    W = sws.reg.AB
    copyto!(view(W, :, 1:D), lds.state_model.A)
    copyto!(view(W, :, D + 1), lds.state_model.b)
    if ux_dim > 0
        copyto!(view(W, :, (D + 2):(D + 1 + ux_dim)), lds.state_model.B)
    end

    # Residual scatter S = dyn_yy - W·dyn_xy - dyn_xy'·W' + W·dyn_xx·W'
    Wxy = sws.elbo.temp                        # D × D scratch (free post-Q_state!)
    mul!(Wxy, W, suf.dyn_xy)

    S_res = sws.reg.Q_sum                      # D × D scratch
    copyto!(S_res, suf.dyn_yy[].mat)
    S_res .-= Wxy
    S_res .-= Wxy'
    #=
    In-place X_A_Xt = W · dyn_xx · W'. Mimic PDMats' X_A_Xt: compute
    `WL = W · L` (where dyn_xx = L·L' via the cached Cholesky) and add
    `WL · WL'` to the upper triangle of S_res via a symmetric rank-k
    BLAS call, then reflect upper → lower so the matrix is EXACTLY
    symmetric and positive-semidefinite by construction. (`mul!(S_res,
    WL, WL', 1, 1)` followed by `Symmetrize!` is *not* equivalent —
    BLAS gemm can produce 1-ULP-asymmetric output, and averaging then
    halves the off-diagonal X_A_Xt contribution.)
    =#
    WL = sws.reg.Sxz                           # (D × dyn_reg_dim) scratch
    #=
    WL = W · L where L is the lower-triangular Cholesky factor of
    dyn_xx. PDMats stores the *upper* factor U in `.chol.factors`
    (uplo='U'); L = U', so the equivalent BLAS call is
    `trmm!(…, 'U', 'T', …)` on the raw factor matrix. This avoids
    the per-call `LowerTriangular(...)` wrapper that
    `mul!(WL, W, chol.L)` would allocate.
    =#
    copyto!(WL, W)
    BLAS.trmm!('R', 'U', 'T', 'N', one(T), suf.dyn_xx[].chol.factors, WL)
    mul!(S_res, WL, transpose(WL), one(T), one(T))

    # MN-prior contribution to the IW posterior scale.
    AB_prior = lds.state_model.AB_prior
    if AB_prior !== nothing
        Wm = W .- AB_prior.M₀
        S_res .+= Wm * AB_prior.Λ * Wm'
    end
    #=
    Reflect upper → lower so the matrix is exactly symmetric. (`mul!`
    of `WL · WL'` above can give 1-ULP-asymmetric output; mirroring
    the upper triangle wins back exact symmetry and preserves the
    mathematically-PSD upper values.)
    =#
    for j in 2:D, i in 1:(j - 1)
        S_res[j, i] = S_res[i, j]
    end

    Q_prior = lds.state_model.Q_prior
    if Q_prior === nothing
        S_res ./= T(suf.dyn_n)
    else
        #=
        iw_map(Ψ, ν, S, N, d) = (Ψ + S) / (ν + N + d + 1), inlined to
        avoid a fresh `(Ψ .+ S)` matrix. `Ψ` is `AbstractMatrix` at the
        type level (IWPrior{T,M<:AbstractMatrix} doesn't pin M on the
        `state_model.Q_prior` field), so we assert the concrete type
        locally to keep the loop type-stable.
        =#
        denom = Q_prior.ν + T(suf.dyn_n) + T(D + 1)
        Ψ = Q_prior.Ψ::Matrix{T}
        for i in eachindex(S_res)
            S_res[i] = (Ψ[i] + S_res[i]) / denom
        end
    end
    copyto!(lds.state_model.Q, S_res)
    return nothing
end
