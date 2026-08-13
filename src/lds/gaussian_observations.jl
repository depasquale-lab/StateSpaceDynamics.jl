#=============================================================================
Gaussian Observations

    Emission kernels: observation_loglikelihood!(cc, dyt, _, lds, x, y, t[, uy])
                      observation_gradient!(out, cc, buf, lds, x, y, t[, uy])
                      observation_hessian!(out, cc, _, _, lds, x, y, t[, α])

    E-Step: Q_obs!(sws, lds, suf)

    M-Step: update_C_d!(lds, suf, sws)
            update_R!(lds, suf, sws)
=============================================================================#

"""
    Q_obs!(ws, lds, E_z, E_zz, y, uy)

Full observation Q-term for Gaussian LDS over all time steps with affine
observation `y_t ~ N(C x_t + d + D uy_t, R)`. `uy` is the per-trial obs-input
matrix `(uy_dim, T_i)`; pass a `0×T_i` matrix when no obs inputs.
"""
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
    uy::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
    C = lds.obs_model.C
    d = lds.obs_model.d
    D_obs = lds.obs_model.D
    uy_dim = size(uy, 1)

    R_U = ws.consts.R_PD.chol.U
    log_det_R = logdet(ws.consts.R_PD)
    const_term = obs_dim * log(T(2π))

    temp = ws.elbo.obs_temp
    work_matrix = ws.elbo.obs_work
    ytil = ws.elbo.ytil
    sum_yy = ws.elbo.sum_yy
    sum_yz = ws.elbo.sum_yz
    work1 = ws.elbo.obs_work1
    work2 = ws.elbo.obs_work2

    fill!(temp, zero(T))

    @views for t in axes(y, 2)
        # Residualize: ytil = y[:,t] - d - D · uy[:,t]
        ytil .= y[:, t] .- d
        if uy_dim > 0
            mul!(ytil, D_obs, uy[:, t], -one(T), one(T))
        end

        mul!(sum_yy, ytil, ytil')

        fill!(sum_yz, zero(T))
        BLAS.ger!(one(T), ytil, E_z[:, t], sum_yz)

        # work_matrix = sum_yy - C·sum_yz' - sum_yz·C' + C·E_zz·C'
        copyto!(work_matrix, sum_yy)
        mul!(work_matrix, C, sum_yz', -one(T), one(T))
        mul!(work1, sum_yz, C')
        work_matrix .-= work1
        mul!(work2, E_zz[:, :, t], C')
        mul!(work_matrix, C, work2, one(T), one(T))

        temp .+= work_matrix
    end

    ldiv!(R_U', temp)
    ldiv!(R_U, temp)
    return T(-0.5) * (tsteps * (const_term + log_det_R) + tr(temp))
end

# Backward-compatible no-input overload.
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    uy = zeros(T, 0, size(y, 2))
    return Q_obs!(ws, lds, E_z, E_zz, y, uy)
end

"""
    Q_obs!(sws, lds, suf)

Total log-likelihood Q-obs term across all trials and time, computed from
the aggregated sufficient statistics in `suf`. Replaces the per-trial,
per-timestep loop of the legacy `Q_obs!(sws, lds, E_z, E_zz, y, uy)`.
"""
function Q_obs!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    uy_dim = lds.uy_dim
    obs_reg_dim = D + 1 + uy_dim
    C = lds.obs_model.C
    d = lds.obs_model.d
    D_obs = lds.obs_model.D

    R_U = sws.consts.R_PD.chol.U
    log_det_R = logdet(sws.consts.R_PD)
    const_term = p * log(T(2π))

    obs_n = suf.obs_n

    # V = [C d D_obs] (p × obs_reg_dim)
    V = view(sws.reg.CD, :, 1:obs_reg_dim)
    copyto!(view(V, :, 1:D), C)
    copyto!(view(V, :, D + 1), d)
    if uy_dim > 0
        copyto!(view(V, :, (D + 2):obs_reg_dim), D_obs)
    end

    # S_obs = obs_yy - V·obs_xy - obs_xy'·V' + V·obs_xx·V'
    S_obs = sws.elbo.obs_temp
    copyto!(S_obs, suf.obs_yy[].mat)
    mul!(S_obs, V, suf.obs_xy, -one(T), one(T))
    mul!(S_obs, transpose(suf.obs_xy), transpose(V), -one(T), one(T))
    V_XX = view(sws.reg.Syz, :, 1:obs_reg_dim)
    mul!(V_XX, V, suf.obs_xx[].mat)
    mul!(S_obs, V_XX, transpose(V), one(T), one(T))

    ldiv!(R_U', S_obs)
    ldiv!(R_U, S_obs)
    return T(-0.5) * (T(obs_n) * (const_term + log_det_R) + tr(S_obs))
end

function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing
    D = lds.latent_dim
    uy_dim = lds.uy_dim
    CD_prior = lds.obs_model.CD_prior

    if CD_prior === nothing
        #=
        Zero-alloc OLS fast path. `sws.reg.Syz` is exactly (p × obs_reg_dim);
        its transpose is the (obs_reg_dim × p) view we ldiv! into. After
        the in-place solve, `sws.reg.Syz` itself holds V = [C d D].
        =#
        Syz_T = transpose(sws.reg.Syz)
        copyto!(Syz_T, suf.obs_xy)
        ldiv!(suf.obs_xx[].chol, Syz_T)
        V = sws.reg.Syz
    else
        V = mn_map(suf.obs_xx[], suf.obs_xy, CD_prior)
    end

    copyto!(lds.obs_model.C, view(V, :, 1:D))
    copyto!(lds.obs_model.d, view(V, :, D + 1))
    if uy_dim > 0
        copyto!(lds.obs_model.D, view(V, :, (D + 2):(D + 1 + uy_dim)))
    end
    return nothing
end

function update_R!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing
    p = lds.obs_dim
    D = lds.latent_dim
    uy_dim = lds.uy_dim

    # sws.reg.CD is exactly (p × obs_reg_dim); no view needed.
    V = sws.reg.CD
    copyto!(view(V, :, 1:D), lds.obs_model.C)
    copyto!(view(V, :, D + 1), lds.obs_model.d)
    if uy_dim > 0
        copyto!(view(V, :, (D + 2):(D + 1 + uy_dim)), lds.obs_model.D)
    end

    # Residual scatter S = obs_yy - V·obs_xy - obs_xy'·V' + V·obs_xx·V'
    Vxy = sws.elbo.obs_temp                    # p × p scratch (free post-Q_obs!)
    mul!(Vxy, V, suf.obs_xy)

    S_res = sws.elbo.obs_work                  # p × p scratch
    copyto!(S_res, suf.obs_yy[].mat)
    S_res .-= Vxy
    S_res .-= Vxy'
    #=
    In-place X_A_Xt = V · obs_xx · V'. Mirror PDMats' X_A_Xt: compute
    `VL = V · L` (obs_xx = L·L' via the cached Cholesky) and add
    `VL · VL'` to the upper triangle via a symmetric rank-k BLAS call,
    then reflect upper → lower for exact symmetry. (`mul!` + `Symmetrize!`
    would halve the off-diagonal contribution because gemm can produce
    1-ULP-asymmetric output that averaging then collapses.)
    =#
    VL = sws.reg.Syz                           # (p × obs_reg_dim) scratch
    #=
    See `update_Q!`: `BLAS.trmm!` on the raw upper-stored
    `chol.factors` (with transa='T' since L = U') avoids the
    `LowerTriangular(...)` wrapper alloc that `mul!(VL, V, chol.L)`
    would do.
    =#
    copyto!(VL, V)
    BLAS.trmm!('R', 'U', 'T', 'N', one(T), suf.obs_xx[].chol.factors, VL)
    mul!(S_res, VL, transpose(VL), one(T), one(T))

    CD_prior = lds.obs_model.CD_prior
    if CD_prior !== nothing
        Wm = V .- CD_prior.M₀
        S_res .+= Wm * CD_prior.Λ * Wm'
    end

    for j in 2:p, i in 1:(j - 1)
        S_res[j, i] = S_res[i, j]
    end

    if lds.obs_model.R_prior === nothing
        S_res ./= T(suf.obs_n)
    else
        Ψ, ν = lds.obs_model.R_prior.Ψ, lds.obs_model.R_prior.ν
        S_res .= iw_map(Ψ, ν, S_res, T(suf.obs_n), p)
    end
    copyto!(lds.obs_model.R, S_res)
    return nothing
end

"""
    observation_loglikelihood!(cc, dyt, _, lds, x, y, t[, uy])

Gaussian emission term: `cR - 0.5*||R^{-1/2}(y_t - Cx_t - d - D uy_t)||^2`.
`dyt` is the `obs_dim` residual scratch; the second buffer is unused.
"""
function observation_loglikelihood!(
    cc::SmoothConstants{T},
    dyt::AbstractVector{T},
    ::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    C = lds.obs_model.C
    d = lds.obs_model.d

    @views mul!(dyt, C, x[:, t])
    if uy !== nothing
        @views mul!(dyt, lds.obs_model.D, uy[:, t], one(T), one(T))
    end
    @views dyt .= y[:, t] .- dyt .- d
    _whiten!(cc.R_PD.chol, dyt)
    return cc.cR - T(0.5) * sum(abs2, dyt)
end

"""
    observation_gradient!(out, cc, buf, lds, x, y, t[, uy])

Gaussian emission gradient: `out = C'R⁻¹ (y_t - Cx_t - d - D uy_t)`, using the
cached `C_inv_R = C'R⁻¹` from `cc`.
"""
function observation_gradient!(
    out::AbstractVector{T},
    cc::SmoothConstants{T},
    buf::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    @views mul!(buf, lds.obs_model.C, x[:, t])
    if uy !== nothing
        @views mul!(buf, lds.obs_model.D, uy[:, t], one(T), one(T))
    end
    @views buf .= y[:, t] .- buf .- lds.obs_model.d
    return mul!(out, cc.C_inv_R, buf)
end

"""
    observation_hessian!(out, cc, _, _, lds, x, y, t[, α, uy])

Gaussian emission curvature: `out .+= α .* (-C'R⁻¹C)`, using the cached
`yt_given_xt = -C'R⁻¹C` from `cc` — constant in `x`, `y`, and any observation
input. Both scratch buffers and `uy` are unused (accepted for interface parity
with the Poisson curvature, which depends on the rate).
"""
function observation_hessian!(
    out::AbstractMatrix{T},
    cc::SmoothConstants{T},
    ::AbstractVector{T},
    ::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    α::T=one(T),
    ::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    @. out += α * cc.yt_given_xt
    return out
end
