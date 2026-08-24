#=============================================================================
Switching LDS (SLDS)

Optional control inputs `ux` (dynamics, `Bₖ u`) and `uy` (observation, `Dₖ v`)
are shared across regimes; the active regime `zₜ` selects which per-regime
`Bₖ` / `Dₖ` multiplies them. `nothing` / zero-row matrices skip the terms.

    Sample:         rand(rng, slds, tsteps; ux, uy)

    Log-Likelihood: joint_loglikelihood!(ws, slds, x, y, w[, ux, uy])

    Gradient:       gradient!(ws, slds, x, y, w[, ux, uy])

    Hessian:        hessian!(ws, slds, x, y, w[, uy])

    Smooth:         smooth!(slds, fs, y, w; x_sample, rng, ux, uy)  # optional joint draw

    E-Step:         estep!(slds, tfs, fb_storage, dl, y, x_samples, slds_ws; ux, uy)

    M-Step:         mstep!(slds, tfs, fb_storage, dl, y, sws; ux, uy)

    Fit:            fit!(slds, y; ux, uy)
=============================================================================#

"""
    _make_slds_fb_storage(dl, seq_ends)

Allocate a single `HMMs.ForwardBackwardStorage` covering all trials. `seq_ends` is the
cumulative timestep index at which each trial ends (HMMs.jl convention). The fb_storage
buffers are sized at `K × sum(T_i)` and `dl.logL` is sized to match.
"""
function _make_slds_fb_storage(
    dl::SLDSDiscreteLayer{T}, seq_ends::AbstractVector{Int}
) where {T}
    total_T = last(seq_ends)
    #=
    HMMs.jl "observations" are just timestep indices into dl.logL; there is no
    control sequence. These are unrelated to the LDS ux / uy
    control-input kwargs.
    =#
    obs_seq = 1:total_T
    control_seq = fill(nothing, total_T)
    return HMMs.initialize_forward_backward(
        dl, obs_seq, control_seq; seq_ends=seq_ends, transition_marginals=true
    )
end

"""
    rand([rng,] slds, tsteps::Integer; ux=nothing, uy=nothing)
    rand([rng,] slds, tsteps_per_trial::AbstractVector{<:Integer}; ux=nothing, uy=nothing)

Sample from a Switching Linear Dynamical System.

- Scalar `tsteps`: returns one trial as `(z::Vector{Int}, x::Matrix, y::Matrix)`.
- Vector of per-trial lengths: returns `(z::Vector{Vector{Int}}, x::Vector{Matrix},
  y::Vector{Matrix})`. Trial lengths may differ.

Optional control inputs (shared across models; the active mode `zₜ` selects
which per-regime `Bₖ` / `Dₖ` multiplies them):
- `ux`: dynamics input consumed by `Bₖ` (`xₜ ~ N(Aₖ xₜ₋₁ + bₖ + Bₖ uₜ₋₁, Qₖ)`).
  Scalar form is an `(ux_dim, tsteps)` matrix; multi-trial is a vector of
  per-trial matrices.
- `uy`: observation input consumed by `Dₖ`. Same shape family as `ux`; required
  when the LDS carry a nonzero-column `D`. Supported for both Gaussian and
  Poisson emissions.
"""
function Random.rand(
    rng::AbstractRNG,
    slds::SLDS{T,S,O},
    tsteps::Integer;
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    lds1 = slds.LDSs[1]
    latent_dim = lds1.latent_dim
    obs_dim = lds1.obs_dim
    Ti = Int(tsteps)

    ux_trial = _check_ux(ux, lds1.ux_dim, Ti, "ux", T)
    uy_trial = _check_uy(uy, lds1.uy_dim, Ti, lds1.obs_model)

    z = Vector{Int}(undef, Ti)
    x = Matrix{T}(undef, latent_dim, Ti)
    y = Matrix{T}(undef, obs_dim, Ti)

    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]

    _sample_slds_trial!(
        rng,
        z,
        x,
        y,
        slds.A,
        slds.πₖ,
        state_params,
        obs_params,
        lds1.obs_model,
        ux_trial,
        uy_trial,
    )

    return z, x, y
end

function Random.rand(
    rng::AbstractRNG,
    slds::SLDS{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    lds1 = slds.LDSs[1]
    latent_dim = lds1.latent_dim
    obs_dim = lds1.obs_dim
    ntrials = length(tsteps_per_trial)

    ux_seq = _normalize_multitrial_ux(ux, lds1.ux_dim, tsteps_per_trial, T, "ux")
    uy_seq = _normalize_multitrial_uy(uy, lds1.uy_dim, tsteps_per_trial, T, lds1.obs_model)

    z = Vector{Vector{Int}}(undef, ntrials)
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)

    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]

    for trial in 1:ntrials
        Ti = Int(tsteps_per_trial[trial])
        z[trial] = Vector{Int}(undef, Ti)
        x[trial] = Matrix{T}(undef, latent_dim, Ti)
        y[trial] = Matrix{T}(undef, obs_dim, Ti)
        _sample_slds_trial!(
            rng,
            z[trial],
            x[trial],
            y[trial],
            slds.A,
            slds.πₖ,
            state_params,
            obs_params,
            lds1.obs_model,
            ux_seq[trial],
            uy_seq[trial],
        )
    end

    return z, x, y
end

function Random.rand(slds::SLDS, tsteps::Integer; kwargs...)
    return rand(Random.default_rng(), slds, tsteps; kwargs...)
end

function Random.rand(slds::SLDS, tsteps_per_trial::AbstractVector{<:Integer}; kwargs...)
    return rand(Random.default_rng(), slds, tsteps_per_trial; kwargs...)
end

# Core SLDS trial sampling logic. `ux_trial` / `uy_trial` are the canonicalized
function _sample_slds_trial!(
    rng,
    z_trial,
    x_trial,
    y_trial,
    A,
    πₖ,
    state_params,
    obs_params,
    obs_model_type,
    ux_trial::AbstractMatrix,
    uy_trial::AbstractMatrix,
)
    tsteps = length(z_trial)
    K = size(A, 1)

    # Sample discrete state sequence using forward sampling
    z_trial[1] = rand(rng, Categorical(πₖ))
    for t in 2:tsteps
        z_trial[t] = rand(rng, Categorical(A[z_trial[t - 1], :]))
    end

    # Sample continuous states and observations given discrete sequence
    return _sample_continuous_given_discrete!(
        rng,
        x_trial,
        y_trial,
        z_trial,
        state_params,
        obs_params,
        obs_model_type,
        ux_trial,
        uy_trial,
    )
end

# Sample continuous dynamics given discrete state sequence
function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::GaussianObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial::AbstractMatrix,
)
    tsteps = length(z_trial)

    # Initial state from the selected LDS
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand(
        rng,
        MvNormal(
            obs_params[k1].C * x_trial[:, 1] +
            obs_params[k1].d +
            obs_params[k1].D * uy_trial[:, 1],
            obs_params[k1].R,
        ),
    )

    # Subsequent states - switch dynamics based on discrete state
    for t in 2:tsteps
        k_curr = z_trial[t]

        # Continuous state follows the current discrete state's dynamics
        # (x_t | x_{t-1}, z_t=k ~ N(A_k x_{t-1} + b_k + B_k u_{t-1}, Q_k),
        # matching `hessian!`)
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] +
                state_params[k_curr].b +
                state_params[k_curr].B * ux_trial[:, t - 1],
                state_params[k_curr].Q,
            ),
        )

        # Observation follows current discrete state's model
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params[k_curr].C * x_trial[:, t] +
                obs_params[k_curr].d +
                obs_params[k_curr].D * uy_trial[:, t],
                obs_params[k_curr].R,
            ),
        )
    end
end

function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::PoissonObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial::AbstractMatrix,
)
    tsteps = length(z_trial)

    # Initial state
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] =
        rand.(
            rng,
            Poisson.(
                exp.(
                    obs_params[k1].C * x_trial[:, 1] +
                    obs_params[k1].d +
                    obs_params[k1].D * uy_trial[:, 1]
                )
            ),
        )

    # Subsequent states
    for t in 2:tsteps
        k_curr = z_trial[t]

        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] +
                state_params[k_curr].b +
                state_params[k_curr].B * ux_trial[:, t - 1],
                state_params[k_curr].Q,
            ),
        )

        y_trial[:, t] =
            rand.(
                rng,
                Poisson.(
                    exp.(
                        obs_params[k_curr].C * x_trial[:, t] +
                        obs_params[k_curr].d +
                        obs_params[k_curr].D * uy_trial[:, t]
                    )
                ),
            )
    end
end

"""
    StatsAPI.fit!(dl::SLDSDiscreteLayer, fb_storage, obs_seq; seq_ends)

Update the discrete transition matrix `dl.A` and initial-state distribution `dl.πₖ`
in place from forward-backward statistics. Mirrors HiddenMarkovModels.jl's
`fit!(::HMM, ...)` pattern using the `ξ[t2]` scratch trick: for each sequence,
`ξ[t2]` is zero by FB convention so it doubles as an accumulator for `sum(ξ[t1:t2-1])`.

Skips fitting observation distributions because the SLDS discrete layer doesn't have
parametric obs distributions; per-state log-likelihoods are filled into `dl.logL`
upstream by the SLDS E-step.
"""
function StatsAPI.fit!(
    dl::SLDSDiscreteLayer{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
) where {T<:Real}
    γ = fb_storage.γ
    ξ = fb_storage.ξ

    # Accumulate ξ[t1:t2-1] into ξ[t2] (zero by FB convention) for each trial.
    tforeach(eachindex(seq_ends)) do k
        # `local`: `t1`/`t2` are also assigned in the sequential loops below,
        # so sharing the bindings would box them (OhMyThreads rejects that).
        local t1, t2
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        scratch = ξ[t2]
        fill!(scratch, zero(eltype(scratch)))
        for t in t1:(t2 - 1)
            scratch .+= ξ[t]
        end
    end

    fill!(dl.πₖ, zero(eltype(dl.πₖ)))
    fill!(dl.A, zero(eltype(dl.A)))
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        dl.πₖ .+= view(γ, :, t1)
        dl.A .+= ξ[t2]
    end

    dl.πₖ ./= sum(dl.πₖ)
    for i in axes(dl.A, 1)
        s = sum(view(dl.A, i, :))
        if s > zero(T)
            dl.A[i, :] ./= s
        end
    end

    return nothing
end

"""
    joint_loglikelihood!(ws, slds, x, y, w[, ux, uy])

Compute weighted complete-data log-likelihood for SLDS.
Returns vector of per-timestep log-likelihoods. `ux` / `uy` are the per-trial
control-input matrices (`nothing` or zero-row skips the `Bₖ u` / `Dₖ v` terms).
"""
function joint_loglikelihood!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},   # K × T responsibilities/weights
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real}
    Tsteps = size(y, 2)

    # Workspace ll_vec may be sized for a longer trial; only touch the active prefix.
    ll_vec = ws.opt.ll_vec
    @views fill!(ll_vec[1:Tsteps], zero(T))

    K = length(slds.LDSs)
    for k in 1:K
        joint_loglikelihood!(
            view(ws.ll_tmp, 1:Tsteps), ws, ws.consts[k], slds.LDSs[k], x, y, ux, uy
        )
        for t in 1:Tsteps
            ll_vec[t] += w[k, t] * ws.ll_tmp[t]
        end
    end

    return view(ll_vec, 1:Tsteps)
end

"""
    gradient!(ws, slds, x, y, w[, ux, uy])

In-place SLDS gradient: each component's complete-data gradient is scaled
per-timestep by the responsibility `w[k, t]` and accumulated. Writes into
`ws.opt.grad_buf` and returns it. `ux` (dynamics input, feeds `-Q⁻¹` /
`A'Q⁻¹` residuals via `Bₖ u`) and `uy` (observation input, feeds the emission
gradient via `Dₖ v`) are per-trial matrices; `nothing` or zero-row skips them.
"""
function gradient!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real}
    latent_dim, Tsteps = size(x)
    K = length(slds.LDSs)

    grad = ws.opt.grad_buf
    fill!(grad, zero(T))

    dxt = ws.opt.dxt
    dxt_next = ws.opt.dxt_next
    obs_buf = ws.opt.dyt
    tmp1 = ws.opt.tmp1
    tmp2 = ws.opt.tmp2
    tmp3 = ws.opt.tmp3

    @views for k in 1:K
        lds_k = slds.LDSs[k]
        cc = ws.consts[k]

        x0 = lds_k.state_model.x0

        A_inv_Q = cc.A_inv_Q          # A'Q^{-1}
        neg_Q_inv = cc.xt_given_xt_1  # -Q^{-1}
        neg_P0_inv = cc.x_t           # -P0^{-1}

        # t = 1: emission + prior, both weighted by w[k,1]
        observation_gradient!(tmp1, cc, obs_buf, lds_k, x, y, 1, uy)
        @. dxt = x[:, 1] - x0
        mul!(tmp3, neg_P0_inv, dxt)
        α = w[k, 1]
        @. grad[:, 1] += α * (tmp1 + tmp3)

        Tsteps == 1 && continue

        # Outgoing dynamics term comes from the factor at time 2, weighted by w[k,2]
        _transition_residual!(dxt_next, lds_k, x, 2, ux)
        mul!(tmp2, A_inv_Q, dxt_next)
        @. grad[:, 1] += w[k, 2] * tmp2

        # 2 .. T-1: emission + incoming factor at t (w[k,t]),
        # outgoing factor at t+1 (w[k,t+1])
        for t in 2:(Tsteps - 1)
            observation_gradient!(tmp1, cc, obs_buf, lds_k, x, y, t, uy)
            _transition_residual!(dxt, lds_k, x, t, ux)
            mul!(tmp3, neg_Q_inv, dxt)
            α = w[k, t]
            @. grad[:, t] += α * (tmp1 + tmp3)

            _transition_residual!(dxt_next, lds_k, x, t + 1, ux)
            mul!(tmp2, A_inv_Q, dxt_next)
            @. grad[:, t] += w[k, t + 1] * tmp2
        end

        # t = T: emission + incoming factor at T, weighted by w[k,T]
        observation_gradient!(tmp1, cc, obs_buf, lds_k, x, y, Tsteps, uy)
        _transition_residual!(dxt, lds_k, x, Tsteps, ux)
        mul!(tmp3, neg_Q_inv, dxt)
        α = w[k, Tsteps]
        @. grad[:, Tsteps] += α * (tmp1 + tmp3)
    end

    return grad
end

"""
    hessian!(ws, slds, x, y, w)

Fill `ws.btd.H_diag`, `ws.btd.H_sub`, `ws.btd.H_super` with the weighted Hessian blocks
for the Laplace/Newton step over `x₁:T` matching Zoltowski et al. Appendix B.

Convention matched:
    x_t | x_{t-1}, z_t=k ~ N(A_k x_{t-1} + b_k, Q_k)
so the dynamics factor that couples (x_{t-1}, x_t) is weighted by w[k,t] = q(z_t=k).

Weights:
- emission curvature at time t uses w[k,t]
- dynamics curvature from factor at time t uses w[k,t]
- off-diagonal block coupling (t-1,t) uses w[k,t]

`uy` (optional observation input) is forwarded to `observation_hessian!`: the
Gaussian curvature ignores it, the Poisson curvature depends on it through the
rate `λ = exp(Cx + d + Dₖ v)`. The state-side blocks never depend on inputs.
"""
function hessian!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real}
    Tsteps = size(x, 2)
    K = length(slds.LDSs)

    H_diag = ws.btd.H_diag
    H_sub = ws.btd.H_sub
    H_super = ws.btd.H_super

    for t in 1:Tsteps
        fill!(H_diag[t], zero(T))
    end
    for t in 1:(Tsteps - 1)
        fill!(H_sub[t], zero(T))
        fill!(H_super[t], zero(T))
    end

    # Two obs_dim scratch vectors for observation_hessian! (Poisson writes the
    # linear predictor and rate into them; Gaussian ignores both).
    z = ws.opt.dyt
    λ = ws.opt.temp_dy

    @views for k in 1:K
        lds_k = slds.LDSs[k]
        cc = ws.consts[k]

        # Cached state-model templates for regime k
        neg_Q_inv = cc.xt_given_xt_1    # -Q^{-1}
        neg_AtQinvA = cc.xt1_given_xt     # -A'Q^{-1}A
        neg_P0_inv = cc.x_t              # -P0^{-1}
        sub_entry = cc.H_sub_entry      #  Q^{-1}A
        super_entry = cc.H_super_entry    # (Q^{-1}A)'

        if Tsteps == 1
            α = w[k, 1]
            @. H_diag[1] += α * neg_P0_inv
            observation_hessian!(H_diag[1], cc, z, λ, lds_k, x, y, 1, α, uy)
            continue
        end

        # Dynamics factor at time t couples (x_{t-1}, x_t), weighted by w[k,t].
        # Off-diagonal blocks between t-1 and t therefore use w[k,t].
        for t in 2:Tsteps
            α = w[k, t]
            @. H_sub[t - 1] += α * sub_entry
            @. H_super[t - 1] += α * super_entry
        end

        # Diagonal state-model contributions:
        # - At t=1: prior term weighted by w[k,1], plus "previous-role" from factor at t=2 weighted by w[k,2]
        @. H_diag[1] += w[k, 1] * neg_P0_inv
        @. H_diag[1] += w[k, 2] * neg_AtQinvA

        # - For 2..T-1: current-role from factor at t (neg_Q_inv) weighted by w[k,t]
        #               previous-role from factor at t+1 (neg_AtQinvA) weighted by w[k,t+1]
        for t in 2:(Tsteps - 1)
            @. H_diag[t] += w[k, t] * neg_Q_inv
            @. H_diag[t] += w[k, t + 1] * neg_AtQinvA
        end

        # - At t=T: current-role from factor at T weighted by w[k,T]
        @. H_diag[Tsteps] += w[k, Tsteps] * neg_Q_inv

        #=
        Emission curvature contributions, weighted by w[k,t]. Shared kernel;
        dispatches on the observation model (Gaussian: cached -C'R⁻¹C,
        Poisson: -C' diag(λ_t) C with λ_t = exp(C x_t + d)).
        =#
        for t in 1:Tsteps
            observation_hessian!(H_diag[t], cc, z, λ, lds_k, x, y, t, w[k, t], uy)
        end
    end

    for t in 1:Tsteps
        Symmetrize!(H_diag[t])
    end

    return nothing
end

function smooth!(
    slds::SLDS{T},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T};
    ws::Union{Nothing,SLDSSmoothWorkspace{T}}=nothing,
    max_iter::Int=20,
    tol::T=T(1e-6),
    linesearch::Union{Nothing,AbstractLineSearch}=BackTrackingLS{T}(),
    x_sample::Union{Nothing,AbstractMatrix{T}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    tsteps = size(y, 2)
    n_active = latent_dim * tsteps

    ws === nothing && (ws = SLDSSmoothWorkspace(T, slds, tsteps))
    btd = ws.btd

    x = fs.x_smooth

    #=
    Warm-start the Newton iteration from the previous EM iteration's smoothed
    mean. If the smoothed mean is all zeros, use the first LDS's prior mean.
    =#
    if all(x .== 0)
        x .= slds.LDSs[1].state_model.x0
    end

    # Active-length views into (possibly) oversized workspace buffers.
    g = view(ws.opt.grad_buf, :, 1:tsteps)
    p = reshape(view(ws.opt.X0, 1:n_active), latent_dim, tsteps)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    ϕ!() = begin
        ll = joint_loglikelihood!(ws, slds, x, y, w, ux, uy)
        return sum(ll)
    end

    compute_grad! = (gcur, xcur) -> begin
        gradient!(ws, slds, xcur, y, w, ux, uy)
        copyto!(gcur, view(ws.opt.grad_buf, :, 1:tsteps))
        return nothing
    end

    build_hess! = (xcur) -> begin
        hessian!(ws, slds, xcur, y, w, uy)
        _negate_blocks!(btd, tsteps)
        return nothing
    end

    solve_dir! =
        (pcur, gcur) -> begin
            gvec = vec(gcur)
            pvec = vec(pcur)
            copyto!(pvec, gvec)
            # SPD path (negated Hessian at MAP).
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
        linesearch;
        max_iter=max_iter,
        tol=tol,
    )

    # Posterior covariances at the MAP via Laplace approx.
    hessian!(ws, slds, x, y, w, uy)
    _negate_blocks!(btd, tsteps)

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    #=
    Optional joint draw from q(x), while `btd` still holds the precision factors.
    `ws.opt.X0` is free after Newton; reuse it for the standard-normal input.
    =#
    if x_sample !== nothing
        z = view(ws.opt.X0, 1:n_active)
        randn!(rng, z)
        block_tridiagonal_sample!(z, btd, tsteps)
        @views x_sample .= fs.x_smooth .+ reshape(z, latent_dim, tsteps)
    end

    @views for t in 1:tsteps
        fs.p_smooth[:, :, t] .= Symmetrize!(fs.p_smooth[:, :, t])
    end

    return fs
end

# Public API wrapper
function smooth(
    slds::SLDS,
    y::AbstractMatrix{T},
    w::AbstractMatrix{T};
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real}
    lds1 = slds.LDSs[1]
    tsteps = size(y, 2)
    ux_m = _check_ux(ux, lds1.ux_dim, tsteps, "ux", T)
    uy_m = _check_uy(uy, lds1.uy_dim, tsteps, lds1.obs_model)
    fs = initialize_FilterSmooth(lds1, tsteps)::FilterSmooth{T}
    smooth!(slds, fs, y, w; ux=ux_m, uy=uy_m)
    return fs.x_smooth, fs.p_smooth
end

"""
    estep!(slds, tfs, fb_storage, dl, y, x_samples, slds_ws; rng, obs_seq, control_seq, seq_ends)

E-step for SLDS using a single sample from the continuous posterior. Updates both
variational posteriors in coordinate-ascent order:

- Fills `dl.logL` (`K × sum(T_i)`) with per-state log-likelihoods from the continuous
  trajectory sampled at the end of the previous smooth (`x_samples`, filled either by the
  `fit!` warm-start or by the prior E-step iteration)
- Updates the discrete posterior q(z) via forward-backward (HiddenMarkovModels.jl, one
  storage covers all trials; HMMs.jl `@threads` across trials internally)
- Updates the continuous posterior q(x) by running the Laplace/Newton smoother on each
  trial with the freshly-updated discrete weights `γ`, filling `tfs[*].x_smooth`,
  `tfs[*].p_smooth`, and `tfs[*].entropy`, and drawing the next joint posterior sample
  into `x_samples[trial]` for the following iteration (an exact draw from q(x) via the
  smoother's precision factors, see `block_tridiagonal_sample!`).

`x_samples` is thus read (to fill `dl.logL`) then overwritten (with the fresh draw) within
each call. `obs_seq`/`control_seq` are the HMMs.jl placeholder sequences built in `fit!`
(timestep indices / `nothing`s) — unrelated to the LDS control-input kwargs
`ux`/`uy`. The latter, when supplied, are per-trial vectors of input matrices
(`ux[trial]` is `(ux_dim, T_trial)`, `uy[trial]` is `(uy_dim, T_trial)`); they
feed the per-regime `Bₖ u` / `Dₖ v` terms of every trial's smoother and
log-likelihood fill. `nothing` (the default) means no inputs.
"""
function estep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_samples::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    rng::AbstractRNG=Random.default_rng(),
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    ntrials = length(y)
    K = length(slds.LDSs)

    # Fill per-trial slices of dl.logL from the previously-sampled trajectory.
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        y_trial = y[trial]
        x_sample = x_samples[trial]
        ux_trial = ux === nothing ? nothing : ux[trial]
        uy_trial = uy === nothing ? nothing : uy[trial]
        for k in 1:K
            ll_view = view(dl.logL, k, t1:t2)
            joint_loglikelihood!(
                ll_view,
                slds_ws,
                slds_ws.consts[k],
                slds.LDSs[k],
                x_sample,
                y_trial,
                ux_trial,
                uy_trial,
            )
        end
    end

    # Update q(z): single batched forward-backward (HMMs.jl threads across trials).
    HMMs.forward_backward!(
        fb_storage, dl, obs_seq, control_seq; seq_ends=seq_ends, transition_marginals=true
    )

    #=
    Update q(x): re-smooth each trial with the new weights γ, and draw the next
    sample into x_samples[trial] on the way out. Overwriting x_samples here is
    fine — the fill loop above already used the previous draw.
    =#
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        w = view(fb_storage.γ, :, t1:t2)  # K × Tsteps
        smooth!(
            slds,
            tfs[trial],
            y[trial],
            w;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
            ux=(ux === nothing ? nothing : ux[trial]),
            uy=(uy === nothing ? nothing : uy[trial]),
        )
    end

    return nothing
end

# tr(A·B) without forming the product: Σ_ij A[i,j]·B[j,i].
@inline function _tr_prod(A::AbstractMatrix, B::AbstractMatrix)
    acc = zero(promote_type(eltype(A), eltype(B)))
    for j in axes(A, 2), i in axes(A, 1)
        acc += A[i, j] * B[j, i]
    end
    return acc
end

"""
    _slds_prior_logdensity(slds)

Sum of the per-regime parameter log-prior contributions (IW on `Q`/`P0`/`R`,
MN on `[A b B]`/`[C d D]`, and the MN-only `[C d]` term for Poisson emissions,
matching the PLDS LBFGS objective). Zero when no priors are set. Needed so the
ELBO tracks the same MAP objective the M-step optimizes; without it the
displayed ELBO can appear non-monotone under priors.
"""
function _slds_prior_logdensity(slds::SLDS{T}) where {T<:Real}
    prior_term = zero(T)
    for lds in slds.LDSs
        sm = lds.state_model
        om = lds.obs_model
        D = lds.latent_dim

        if sm.Q_prior !== nothing
            prior_term += iw_logprior_term(sm.Q, sm.Q_prior)
        end
        if sm.P0_prior !== nothing
            prior_term += iw_logprior_term(sm.P0, sm.P0_prior)
        end
        if sm.x0_prior !== nothing
            prior_term += mn_logprior_term(reshape(sm.x0, :, 1), sm.P0, sm.x0_prior)
        end
        if sm.AB_prior !== nothing
            ux_dim = lds.ux_dim
            W_ab = Matrix{T}(undef, D, D + 1 + ux_dim)
            @views W_ab[:, 1:D] .= sm.A
            @views W_ab[:, D + 1] .= sm.b
            ux_dim > 0 && (@views W_ab[:, (D + 2):end] .= sm.B)
            prior_term += mn_logprior_term(W_ab, sm.Q, sm.AB_prior)
        end

        if om isa GaussianObservationModel{T}
            if om.R_prior !== nothing
                prior_term += iw_logprior_term(om.R, om.R_prior)
            end
            if om.CD_prior !== nothing
                uy_dim = lds.uy_dim
                W_cd = Matrix{T}(undef, lds.obs_dim, D + 1 + uy_dim)
                @views W_cd[:, 1:D] .= om.C
                @views W_cd[:, D + 1] .= om.d
                uy_dim > 0 && (@views W_cd[:, (D + 2):end] .= om.D)
                prior_term += mn_logprior_term(W_cd, om.R, om.CD_prior)
            end
        elseif om isa PoissonObservationModel{T}
            if om.CD_prior !== nothing
                W_cd = Matrix{T}(undef, lds.obs_dim, D + 1)
                @views W_cd[:, 1:D] .= om.C
                @views W_cd[:, D + 1] .= om.d
                Wm = W_cd .- om.CD_prior.M₀
                prior_term -= T(0.5) * sum(Wm .* (Wm * om.CD_prior.Λ))
            end
        end
    end
    return prior_term
end

"""
    _slds_trial_elbo(slds, fs, fb_storage, y_trial, slds_ws, t1, t2, ux_trial, uy_trial)

One trial's contribution to the SLDS ELBO (everything except the parameter
log-prior). Split out of `elbo!` so a caller can evaluate a trial against a
different parameter set than its neighbours, after refreshing the regime
constants.

Assumes `slds_ws.consts` already holds the constants of `slds`.
"""
function _slds_trial_elbo(
    slds::SLDS{T,S,O},
    fs::FilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y_trial::AbstractMatrix{T},
    slds_ws::SLDSSmoothWorkspace{T},
    t1::Int,
    t2::Int,
    ux_trial::Union{Nothing,AbstractMatrix{T}},
    uy_trial::Union{Nothing,AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    K = length(slds.LDSs)
    Tsteps = t2 - t1 + 1
    w = view(fb_storage.γ, :, t1:t2)  # K × Tsteps

    trial_elbo = zero(T)
    x_smooth_trial = fs.x_smooth

    # E_q[log p(y, x | z)], plug-in at the posterior mean, weighted by γ.
    for k in 1:K
        ll = view(slds_ws.ll_tmp, 1:Tsteps)::AbstractVector{T}
        joint_loglikelihood!(
            ll,
            slds_ws,
            slds_ws.consts[k],
            slds.LDSs[k],
            x_smooth_trial,
            y_trial,
            ux_trial,
            uy_trial,
        )
        for t in 1:Tsteps
            trial_elbo += w[k, t] * ll[t]
        end
    end

    #=
    ½ tr(H Σ) covariance correction. H = weighted Hessian (hessian! writes
    it un-negated into slds_ws.btd); Σ = p_smooth on the diagonal,
    p_smooth_tt1[:,:,t] = Cov(x_t, x_{t-1}) off it. Sum both off-diagonal
    traces rather than doubling one — don't assume exact block symmetry.
    =#
    hessian!(slds_ws, slds, x_smooth_trial, y_trial, w, uy_trial)
    H_diag = slds_ws.btd.H_diag
    H_sub = slds_ws.btd.H_sub
    H_super = slds_ws.btd.H_super
    for t in 1:Tsteps
        trial_elbo += T(0.5) * _tr_prod(H_diag[t], view(fs.p_smooth, :, :, t))
    end
    for t in 2:Tsteps
        Σ_ttm1 = view(fs.p_smooth_tt1, :, :, t)  # Cov(x_t, x_{t-1})
        trial_elbo += T(0.5) * _tr_prod(H_super[t - 1], Σ_ttm1)
        trial_elbo += T(0.5) * _tr_prod(H_sub[t - 1], transpose(Σ_ttm1))
    end

    # E_q[log p(z_1)].
    for k in 1:K
        trial_elbo += w[k, 1] * log(slds.πₖ[k] + T(1e-12))
    end

    #=
    E_q[log p(z_t | z_{t-1})] = Σ_t Σ_ij ξ_t[i,j] log A[i,j]. ξ is global-
    indexed; ξ[t2] is zero by FB convention, so iterate t1..t2-1.
    =#
    for t in t1:(t2 - 1)
        ξt = fb_storage.ξ[t]
        for i in 1:K, j in 1:K
            trial_elbo += ξt[i, j] * log(slds.A[i, j] + T(1e-12))
        end
    end

    # + H[q(x)] (filled by `smooth!` from the BT log-determinant).
    trial_elbo += fs.entropy

    #=
    + H[q(z)], the FB chain entropy
    −Σ_k γ₁ log γ₁ − Σ_t Σ_ij ξ_t[i,j] (log ξ_t[i,j] − log γ_t[i]).
    ξ_t[i,j] > 0 ⇒ γ_t[i] > 0, so both logs are safe.
    =#
    for k in 1:K
        wk1 = w[k, 1]
        wk1 > 0 && (trial_elbo -= wk1 * log(wk1))
    end
    for t in t1:(t2 - 1)
        ξt = fb_storage.ξ[t]
        tloc = t - t1 + 1
        for i in 1:K, j in 1:K
            ξij = ξt[i, j]
            ξij > 0 && (trial_elbo -= ξij * (log(ξij) - log(w[i, tloc])))
        end
    end

    return trial_elbo
end

"""
    elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends)

Evidence lower bound for the SLDS at the current variational posteriors —
q(x) the per-trial joint Gaussian from the Laplace smoother, q(z) the
forward-backward chain posterior:

    ELBO = E_q[log p(y, x | z)] + E_q[log p(z)] + H[q(x)] + H[q(z)] + log p(θ)

- `E_q[log p(y, x | z)]` is the responsibility-weighted log-density at the
  posterior mean plus the covariance correction `½ tr(H Σ)`, where `H` is the
  weighted Hessian over `x₁:T` and `Σ` the block-tridiagonal posterior
  covariance. Exact for Gaussian emissions (the weighted log-density is
  quadratic in `x`); the standard second-order/Laplace approximation for
  Poisson.
- `E_q[log p(z)]` uses the FB marginals `γ` (initial) and pairwise `ξ`
  (transitions).
- `H[q(z)]` is the Markov-chain entropy of the FB posterior,
  `−Σ γ₁ log γ₁ − Σ_t Σ_ij ξ_t(i,j) log(ξ_t(i,j)/γ_t(i))` — not the
  factorized `−Σ γ log γ`, which would overstate the entropy of a chain.
- `log p(θ)` collects per-regime IW/MN prior log-densities so the ELBO tracks
  the MAP objective the M-step optimizes (zero when no priors are set).

The continuous term is evaluated at the smoothed mean (deterministic given the
current posteriors), not at the E-step's posterior sample. For K = 1 with
Gaussian emissions and no priors this equals the exact marginal log-likelihood.

Returns a scalar. Overwrites `slds_ws.btd`'s Hessian blocks and `ll_tmp`.
"""
function elbo!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    total_elbo = zero(T)
    ntrials = length(y)

    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        total_elbo += _slds_trial_elbo(
            slds,
            tfs[trial],
            fb_storage,
            y[trial],
            slds_ws,
            t1,
            t2,
            ux === nothing ? nothing : ux[trial],
            uy === nothing ? nothing : uy[trial],
        )
    end

    return total_elbo + _slds_prior_logdensity(slds)
end

"""
    elbo(slds, y; ux=nothing, uy=nothing, rng=Random.default_rng())

Evidence lower bound of an `SLDS` at the current parameters (allocating
convenience wrapper around the workspace-based [`elbo!`](@ref)): warm-starts
the continuous posterior with uniform discrete weights, runs one variational
E-step (forward-backward for `q(z)`, Laplace smoothing for `q(x)`), and
evaluates the ELBO at the resulting posteriors.

The E-step consumes a joint sample from `q(x)` to build the discrete-layer
log-likelihoods, so the returned value is **stochastic** — pass `rng` for
reproducibility. This matches the first entry of the ELBO trace returned by
`fit!` when given the same `rng`.

# Arguments
- `y`: observations — a `(obs_dim, T)` matrix, a `(obs_dim, T, ntrials)`
  array, or a `Vector{<:AbstractMatrix}` of per-trial `(obs_dim, T_i)`
  matrices (ragged lengths allowed).
- `ux` / `uy`: optional control inputs in the same shape family as `y`
  (`nothing` when the regimes carry no `B` / `D`). See [`fit!`](@ref).

Returns a scalar.
"""
function elbo(
    slds::SLDS{T,S,O},
    y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}};
    ux=nothing,
    uy=nothing,
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    # `Data` centralizes shape validation and canonicalizes the three
    # observation/input forms (regime dims are uniform, so validating against
    # LDSs[1] covers all regimes). Absent ux/uy become zero-row matrices.
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    y_seq = data.y
    ux_seq = data.ux
    uy_seq = data.uy

    K = length(slds.LDSs)
    ntrials = length(y_seq)
    seq_ends = cumsum(data.tsteps)
    total_T = last(seq_ends)
    T_max = maximum(data.tsteps)

    tfs = initialize_FilterSmooth(slds.LDSs[1], data.tsteps)::TrialFilterSmooth{T}
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))
    fb_storage = _make_slds_fb_storage(dl, seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)
    slds_ws = SLDSSmoothWorkspace(T, slds, T_max)
    x_samples = [Matrix{T}(undef, slds.LDSs[1].latent_dim, Ti) for Ti in data.tsteps]

    # Warm-start q(x) with uniform weights, drawing the sample the E-step's
    # discrete update consumes (mirrors the fit! warm-start).
    for trial in 1:ntrials
        w_uniform = fill(one(T) / K, K, data.tsteps[trial])
        smooth!(
            slds,
            tfs[trial],
            y_seq[trial],
            w_uniform;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
            ux=ux_seq[trial],
            uy=uy_seq[trial],
        )
    end

    estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y_seq,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux_seq,
        uy=uy_seq,
    )

    return elbo!(
        slds, tfs, fb_storage, y_seq, slds_ws; seq_ends=seq_ends, ux=ux_seq, uy=uy_seq
    )
end

"""
    loglikelihood(slds, y)

Marginal (observed-data) log-likelihood for an SLDS — **not implemented**.

The marginal `log p(y)` requires summing over all `K^T` discrete regime
sequences (the switching model has no closed-form filter). Use
[`elbo`](@ref)`(slds, y)` for a variational lower bound on `log p(y)`, or the
ELBO trace returned by `fit!`.
"""
function StatsAPI.loglikelihood(slds::SLDS, y)
    return error(
        "marginal loglikelihood is not implemented for the SLDS (marginalizing the " *
        "discrete regime sequence requires summing over K^T paths). Use " *
        "elbo(slds, y) for a variational lower bound, or the ELBO trace from fit!.",
    )
end

"""
    mstep!(slds, tfs, fb_storage, y, sws; obs_seq, seq_ends, ux=nothing, uy=nothing)

M-step for SLDS.

- Updates discrete parameters (`slds.A`, `slds.πₖ`) via `StatsAPI.fit!` on the discrete
  layer (uses HMMs.jl's `ξ[t2]` scratch trick).
- Updates each LDS component using γ-weighted sufficient statistics aggregated
  by `_aggregate_td_suff_stats_weighted!`. For Gaussian sub-LDSs this is the
  full suf-based M-step (regression + IW MAP). For Poisson sub-LDSs the state-
  side updates flow through the same suf path; the emission [C d] is updated
  via the existing LBFGS routine (Poisson is non-conjugate and cannot be
  folded into the regression).

`ux` / `uy` are the per-trial control-input sequences; when present, the weighted
aggregator folds `Bₖ u` / `Dₖ v` into the regression targets so `Bₖ` (Gaussian
and Poisson dynamics) and `Dₖ` (Gaussian emission, and Poisson emission via the
LBFGS routine) are re-estimated alongside `Aₖ` / `Cₖ`.
"""
function mstep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T};
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    ntrials = length(y)

    # Discrete-layer M-step (slds.A, slds.πₖ are updated in place via dl).
    StatsAPI.fit!(dl, fb_storage, obs_seq; seq_ends=seq_ends)

    #=
    `Data` canonicalizes absent ux/uy to zero-row matrices and validates the
    supplied ones. All regimes share the same input dims (enforced by
    `validate_SLDS`), so one `Data` serves every `lds_k`.
    =#
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)

    # One reusable SufficientStatistics; overwritten per regime by the
    # weighted aggregator.
    suf = _initialize_td_sufficient_statistics(T, slds.LDSs[1], data.tsteps)

    #=
    x0/P0 are tied across modes. Since the smoother gives one q(x) per trial
    and Σₖ γₖ(t=1) = 1, the pooled unit-weight init stats are exactly the sum
    over modes of the per-mode init stats the aggregator already computes —
    so accumulate them across the loop instead of recomputing.
    =#
    D = slds.LDSs[1].latent_dim
    init_xy = zeros(T, 1, D)
    init_yy = zeros(T, D, D)

    weights = Vector{AbstractVector{T}}(undef, ntrials)
    for k in 1:K
        lds_k = slds.LDSs[k]
        for trial in 1:ntrials
            t1, t2 = HMMs.seq_limits(seq_ends, trial)
            weights[trial] = view(fb_storage.γ, k, t1:t2)
        end

        _aggregate_td_suff_stats_weighted!(suf, tfs, lds_k, data, weights, sws)
        init_xy .+= suf.init_xy
        init_yy .+= suf.init_yy[]

        # Per-regime updates cover only dynamics + emissions (init is tied).
        if lds_k.obs_model isa GaussianObservationModel{T}
            update_A_b!(lds_k, suf, sws)
            update_Q!(lds_k, suf, sws)
            update_C_d!(lds_k, suf, sws)
            update_R!(lds_k, suf, sws)
        elseif lds_k.obs_model isa PoissonObservationModel{T}
            update_A_b!(lds_k, suf, sws)
            update_Q!(lds_k, suf, sws)
            # Single sws wrapped as a pool of one; maybe thread in future
            update_observation_model!(lds_k, tfs, y, [sws], weights; uy=data.uy)
        else
            throw(ArgumentError("Unsupported observation model $(typeof(lds_k.obs_model))"))
        end
    end

    # Fit the single shared x0/P0 from the pooled stats, then broadcast.
    copyto!(suf.init_xy, init_xy)
    copyto!(suf.init_yy[], init_yy)
    suf.init_n = T(ntrials)
    _update_shared_initial_state!(slds, suf, sws)

    return nothing
end

"""
    _update_shared_initial_state!(slds, suf, sws)

Fit the single initial-state distribution `N(x0, P0)` shared by all SLDS modes
from the pooled init stats in `suf` (see `mstep!`) and copy it into every
`state_model`.
"""
function _update_shared_initial_state!(
    slds::SLDS{T}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real}
    lds1 = deepcopy(slds.LDSs[1])
    update_initial_state_mean!(lds1, suf)
    update_initial_state_covariance!(lds1, suf, sws)
    fit_x0, fit_P0 = lds1.fit_bool[1], lds1.fit_bool[2]
    for k in eachindex(slds.LDSs)
        fit_x0 && copyto!(slds.LDSs[k].state_model.x0, lds1.state_model.x0)
        fit_P0 && copyto!(slds.LDSs[k].state_model.P0, lds1.state_model.P0)
    end
    return nothing
end

"""
    fit!(slds::SLDS, y; ux=nothing, uy=nothing, max_iter=50, progress=true)

Fit SLDS using variational Laplace EM. Runs for exactly `max_iter` iterations
(no early-stopping criterion: the E-step's posterior sampling makes the ELBO
trace noisy across iterations, so a tolerance check on successive differences
would fire spuriously). Returns the per-iteration ELBO trace.

`y` is a single trial `(obs_dim × T)` matrix, a `(obs_dim, T, ntrials)` array,
or a vector of per-trial matrices (ragged `T_i` allowed). Internally a single
batched `HMMs.ForwardBackwardStorage` of length `sum(T_i)` is allocated, with
`seq_ends = cumsum(T_i)` to demarcate trials.

Optional control inputs `ux` / `uy` accept the same shape family as `y`
(`nothing` when the regimes carry no `B` / `D`). They are shared across regimes
— the active regime `zₜ` selects which per-regime `Bₖ` / `Dₖ` multiplies the
input — and are re-estimated per regime alongside the other parameters. The
input dimensions must match across regimes (enforced by `validate_SLDS`).
"""
function fit!(
    slds::SLDS{T,S,O},
    y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}};
    ux=nothing,
    uy=nothing,
    max_iter::Int=50,
    progress::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    #=
    `Data` centralizes shape validation and canonicalizes the three
    observation/input forms (regime dims are uniform, so validating against
    LDSs[1] covers all regimes). Absent ux/uy become zero-row matrices.
    =#
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    y_seq = data.y
    ux_seq = data.ux
    uy_seq = data.uy

    K = length(slds.LDSs)
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    tsteps_per_trial = data.tsteps
    ntrials = length(y_seq)
    seq_ends = cumsum(tsteps_per_trial)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_per_trial)

    # Continuous-state smoother storage (per-trial sized).
    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_per_trial)::TrialFilterSmooth{T}

    # Discrete-layer wrapper (logL sized for the batched timestep sequence).
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))

    # Single batched fb_storage covering all trials.
    fb_storage = _make_slds_fb_storage(dl, seq_ends)

    # Cached batched HMMs.jl placeholder sequences (timestep indices / nothings).
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    # Workspaces — allocated once at max trial length, reused each iteration.
    # `sws` sizes its regression buffers for the (uniform) input dims so the
    # weighted aggregator can fit `[Aₖ bₖ Bₖ]` / `[Cₖ dₖ Dₖ]`.
    sws = SmoothWorkspace(
        T,
        latent_dim,
        obs_dim,
        T_max;
        ux_dim=slds.LDSs[1].ux_dim,
        uy_dim=slds.LDSs[1].uy_dim,
    )
    slds_ws = SLDSSmoothWorkspace(T, slds, T_max)
    x_samples = [Matrix{T}(undef, latent_dim, Ti) for Ti in tsteps_per_trial]

    prog = if progress
        Progress(max_iter; desc="Fitting SLDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end
    elbos = Vector{T}(undef, max_iter)

    #=
    Warm-start: smooth each trial once with uniform weights, drawing the first
    sample into x_samples for the first E-step to consume.
    =#
    for trial in 1:ntrials
        Ti = tsteps_per_trial[trial]
        w_uniform = fill(one(T) / K, K, Ti)
        smooth!(
            slds,
            tfs[trial],
            y_seq[trial],
            w_uniform;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
            ux=ux_seq[trial],
            uy=uy_seq[trial],
        )
    end

    for iter in 1:max_iter

        #=
        E-step: fill q(z) from the current samples, run forward-backward,
        re-smooth q(x), and draw the next samples for the following iteration.
        =#
        estep!(
            slds,
            tfs,
            fb_storage,
            dl,
            y_seq,
            x_samples,
            slds_ws;
            rng=rng,
            obs_seq=obs_seq,
            control_seq=control_seq,
            seq_ends=seq_ends,
            ux=ux_seq,
            uy=uy_seq,
        )

        # Compute the ELBO at the current posteriors.
        elbos[iter] = elbo!(
            slds, tfs, fb_storage, y_seq, slds_ws; seq_ends=seq_ends, ux=ux_seq, uy=uy_seq
        )

        # M-step: update discrete and continuous parameters.
        mstep!(
            slds,
            tfs,
            fb_storage,
            dl,
            y_seq,
            sws;
            obs_seq=obs_seq,
            seq_ends=seq_ends,
            ux=ux_seq,
            uy=uy_seq,
        )
        refresh_slds_constants!(slds_ws, slds)

        prog !== nothing && next!(prog)
    end

    if prog !== nothing
        finish!(prog)
    end
    return elbos
end
