"""
    Q_state!(ws, lds, E_z, E_zz, E_zz_prev, u)

State Q-term for an LDS with affine dynamics `x_t ~ N(A x_{t-1} + b + B u_{t-1}, Q)`.
In-place version of `Q_state` that uses pre-allocated buffers from `SmoothWorkspace`.
Uses cached Cholesky factors from `compute_smooth_constants!`. `u` is the per-trial
dynamics-control matrix `(u_dim, T_i)`; pass a `0×T_i` matrix when no inputs.
"""
function Q_state!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    u::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tstep = size(E_z, 2)
    D = lds.latent_dim
    u_dim = size(u, 1)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0

    # Use cached Cholesky factors (already computed by compute_smooth_constants!)
    Q_U = ws.Q_PD[].chol.U
    P0_U = ws.P0_PD[].chol.U

    log_det_Q = logdet(ws.Q_PD[])
    log_det_P0 = logdet(ws.P0_PD[])

    temp = ws.elbo_temp
    sum_E_zz = ws.elbo_sum_E_zz
    sum_E_zzm1 = ws.elbo_sum_E_zzm1
    sum_E_cross = ws.elbo_sum_E_cross
    sum_mu_t = ws.elbo_sum_mu_t
    sum_mu_tm1 = ws.elbo_sum_mu_tm1
    temp2 = ws.elbo_temp2

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

    # Input-specific accumulators (only allocated when u_dim > 0). Allocating
    # 0-element arrays here would still cost an `Array` struct each call,
    # which adds up to thousands of trivial allocations across a fit.
    has_input = u_dim > 0
    sum_u = has_input ? zeros(T, u_dim) : Vector{T}()
    sum_mu_t_u = has_input ? zeros(T, D, u_dim) : Matrix{T}(undef, 0, 0)
    sum_mu_tm1_u = has_input ? zeros(T, D, u_dim) : Matrix{T}(undef, 0, 0)
    sum_uu = has_input ? zeros(T, u_dim, u_dim) : Matrix{T}(undef, 0, 0)

    @views for t in 2:tstep
        sum_E_zz .+= E_zz[:, :, t]
        sum_E_zzm1 .+= E_zz[:, :, t - 1]
        sum_E_cross .+= E_zz_prev[:, :, t]
        sum_mu_t .+= E_z[:, t]
        sum_mu_tm1 .+= E_z[:, t - 1]

        if has_input
            u_tm1 = u[:, t - 1]
            sum_u .+= u_tm1
            BLAS.ger!(one(T), E_z[:, t], u_tm1, sum_mu_t_u)
            BLAS.ger!(one(T), E_z[:, t - 1], u_tm1, sum_mu_tm1_u)
            BLAS.ger!(one(T), u_tm1, u_tm1, sum_uu)
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
    mul!(ws.tmp1, A, sum_mu_tm1)
    mul!(temp, ws.tmp1, b', one(T), one(T))
    mul!(temp, b, ws.tmp1', one(T), one(T))
    mul!(temp, b, b', T(tstep - 1), one(T))

    # Input cross terms (`Bu_{t-1} := B u_{t-1}`). All terms here are
    # contributions to `Σ_t E[(x_t - A x_{t-1} - b - B u_{t-1})(...)']` that
    # involve at least one `B u_{t-1}` factor.
    if u_dim > 0
        # -= sum_mu_t_u · B'  and  -= B · sum_mu_t_u'
        mul!(temp, sum_mu_t_u, B', -one(T), one(T))
        mul!(temp, B, sum_mu_t_u', -one(T), one(T))
        # += (A · sum_mu_tm1_u) · B'  and  += B · (A · sum_mu_tm1_u)'
        # Intermediate has shape (D × u_dim); no fixed-size workspace buffer.
        A_sumXU = A * sum_mu_tm1_u
        mul!(temp, A_sumXU, B', one(T), one(T))
        mul!(temp, B, A_sumXU', one(T), one(T))
        # += b · (B · sum_u)'  and  += (B · sum_u) · b'
        B_sumu = B * sum_u  # D-vector
        mul!(temp, reshape(b, :, 1), reshape(B_sumu, 1, :), one(T), one(T))
        mul!(temp, reshape(B_sumu, :, 1), reshape(b, 1, :), one(T), one(T))
        # += B · sum_uu · B'
        B_sumuu = B * sum_uu  # D × u_dim
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
    u = zeros(T, 0, size(E_z, 2))
    return Q_state!(ws, lds, E_z, E_zz, E_zz_prev, u)
end

"""
    Q_state!(sws, lds, suf)

Total log-likelihood Q-state term across all trials, computed from the
aggregated sufficient statistics in `suf`. Replaces the per-trial,
per-timestep loops of the legacy `Q_state!(sws, lds, E_z, E_zz, E_zz_prev, u)`.

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
    u_dim = lds.state_input_dim
    dyn_reg_dim = D + 1 + u_dim

    Q_U = sws.Q_PD[].chol.U
    P0_U = sws.P0_PD[].chol.U

    log_det_Q = logdet(sws.Q_PD[])
    log_det_P0 = logdet(sws.P0_PD[])

    N = suf.init_n
    dyn_n = suf.dyn_n

    log2π = log(T(2π))
    const_init = T(N) * D * log2π
    const_trans = T(dyn_n) * D * log2π

    # S_init = init_yy - μ x0' - x0 μ' + N x0 x0'    (μ = Σ x_init)
    S_init = sws.elbo_temp
    copyto!(S_init, suf.init_yy[].mat)
    μ_sum = vec(suf.init_xy)
    BLAS.ger!(-one(T), μ_sum, x0, S_init)
    BLAS.ger!(-one(T), x0, μ_sum, S_init)
    BLAS.ger!(T(N), x0, x0, S_init)

    ldiv!(P0_U', S_init)
    ldiv!(P0_U, S_init)
    Q_val = T(-0.5) * (const_init + T(N) * log_det_P0 + tr(S_init))

    # W = [A b B] (D × dyn_reg_dim)
    W = view(sws.AB, :, 1:dyn_reg_dim)
    copyto!(view(W, :, 1:D), A)
    copyto!(view(W, :, D + 1), b)
    if u_dim > 0
        copyto!(view(W, :, (D + 2):dyn_reg_dim), B)
    end

    S_trans = sws.elbo_temp                 # reuse (S_init no longer needed)
    copyto!(S_trans, suf.dyn_yy[].mat)
    mul!(S_trans, W, suf.dyn_xy, -one(T), one(T))
    mul!(S_trans, transpose(suf.dyn_xy), transpose(W), -one(T), one(T))
    # S_trans += W · dyn_xx · W'
    W_XX = view(sws.Sxz, :, 1:dyn_reg_dim)
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

    S0 = sws.S0_sum                                  # D × D scratch
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
    u_dim = lds.state_input_dim
    AB_prior = lds.state_model.AB_prior

    if AB_prior === nothing
        # Zero-alloc OLS fast path. `sws.Sxz` is exactly (D × dyn_reg_dim);
        # its transpose is the (dyn_reg_dim × D) view we ldiv! into. After
        # the in-place solve, `sws.Sxz` itself holds the transposed solution
        # `transpose(dyn_xx \ dyn_xy)` = the W = [A b B] regression matrix.
        Sxz_T = transpose(sws.Sxz)
        copyto!(Sxz_T, suf.dyn_xy)
        ldiv!(suf.dyn_xx[].chol, Sxz_T)
        W = sws.Sxz
    else
        # MN-prior MAP path — keep `mn_map` (allocates) for now.
        W = mn_map(suf.dyn_xx[], suf.dyn_xy, AB_prior)
    end

    copyto!(lds.state_model.A, view(W, :, 1:D))
    copyto!(lds.state_model.b, view(W, :, D + 1))
    if u_dim > 0
        copyto!(lds.state_model.B, view(W, :, (D + 2):(D + 1 + u_dim)))
    end
    return nothing
end

function update_Q!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing
    D = lds.latent_dim
    u_dim = lds.state_input_dim

    # sws.AB is exactly (D × dyn_reg_dim); no view needed.
    W = sws.AB
    copyto!(view(W, :, 1:D), lds.state_model.A)
    copyto!(view(W, :, D + 1), lds.state_model.b)
    if u_dim > 0
        copyto!(view(W, :, (D + 2):(D + 1 + u_dim)), lds.state_model.B)
    end

    # Residual scatter S = dyn_yy - W·dyn_xy - dyn_xy'·W' + W·dyn_xx·W'
    Wxy = sws.elbo_temp                        # D × D scratch (free post-Q_state!)
    mul!(Wxy, W, suf.dyn_xy)

    S_res = sws.Q_sum                          # D × D scratch
    copyto!(S_res, suf.dyn_yy[].mat)
    S_res .-= Wxy
    S_res .-= Wxy'
    # In-place X_A_Xt = W · dyn_xx · W'. Mimic PDMats' X_A_Xt: compute
    # `WL = W · L` (where dyn_xx = L·L' via the cached Cholesky) and add
    # `WL · WL'` to the upper triangle of S_res via a symmetric rank-k
    # BLAS call, then reflect upper → lower so the matrix is EXACTLY
    # symmetric and positive-semidefinite by construction. (`mul!(S_res,
    # WL, WL', 1, 1)` followed by `Symmetrize!` is *not* equivalent —
    # BLAS gemm can produce 1-ULP-asymmetric output, and averaging then
    # halves the off-diagonal X_A_Xt contribution.)
    WL = sws.Sxz                               # (D × dyn_reg_dim) scratch
    # WL = W · L where L is the lower-triangular Cholesky factor of
    # dyn_xx. PDMats stores the *upper* factor U in `.chol.factors`
    # (uplo='U'); L = U', so the equivalent BLAS call is
    # `trmm!(…, 'U', 'T', …)` on the raw factor matrix. This avoids
    # the per-call `LowerTriangular(...)` wrapper that
    # `mul!(WL, W, chol.L)` would allocate.
    copyto!(WL, W)
    BLAS.trmm!('R', 'U', 'T', 'N', one(T), suf.dyn_xx[].chol.factors, WL)
    mul!(S_res, WL, transpose(WL), one(T), one(T))

    # MN-prior contribution to the IW posterior scale.
    AB_prior = lds.state_model.AB_prior
    if AB_prior !== nothing
        Wm = W .- AB_prior.M₀
        S_res .+= Wm * AB_prior.Λ * Wm'
    end
    # Reflect upper → lower so the matrix is exactly symmetric. (`mul!`
    # of `WL · WL'` above can give 1-ULP-asymmetric output; mirroring
    # the upper triangle wins back exact symmetry and preserves the
    # mathematically-PSD upper values.)
    for j in 2:D, i in 1:(j - 1)
        S_res[j, i] = S_res[i, j]
    end

    Q_prior = lds.state_model.Q_prior
    if Q_prior === nothing
        S_res ./= T(suf.dyn_n)
    else
        # iw_map(Ψ, ν, S, N, d) = (Ψ + S) / (ν + N + d + 1), inlined to
        # avoid a fresh `(Ψ .+ S)` matrix. `Ψ` is `AbstractMatrix` at the
        # type level (IWPrior{T,M<:AbstractMatrix} doesn't pin M on the
        # `state_model.Q_prior` field), so we assert the concrete type
        # locally to keep the loop type-stable.
        denom = Q_prior.ν + T(suf.dyn_n) + T(D + 1)
        Ψ = Q_prior.Ψ::Matrix{T}
        for i in eachindex(S_res)
            S_res[i] = (Ψ[i] + S_res[i]) / denom
        end
    end
    copyto!(lds.state_model.Q, S_res)
    return nothing
end
