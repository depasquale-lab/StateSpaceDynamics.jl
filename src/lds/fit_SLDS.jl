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

    Fit:            fit!(slds, y; ux, uy, smoothing_iters)

    Infer:          smooth(slds, y; ux, uy, smoothing_iters, tol)  # -> (; x, γ, elbo, p)

    ELBO:           elbo(slds, y; ux, uy) == loglikelihood(slds, y; ux, uy)
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
    depends_on::Union{Nothing,NamedTuple}=nothing,
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

    if depends_on === nothing && _has_parameter_dependence(lds1)
        _single_trial_group_error("slds")
    end
    grp = _slds_parameter_grouping(slds, 1; depends_on=depends_on)
    regimes = if grp === nothing
        slds.LDSs
    else
        _slds_cell_sldss(slds, grp)[grp.trial_cell[1]].LDSs
    end

    state_params = [_extract_state_params(lds.state_model) for lds in regimes]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in regimes]

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
    depends_on::Union{Nothing,NamedTuple}=nothing,
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

    #=
    Per-trial, per-regime parameter sets: one entry per trial, each a vector
    over regimes. Ungrouped, every trial shares the same vector.
    =#
    grp = _slds_parameter_grouping(slds, ntrials; depends_on=depends_on)
    if grp === nothing
        base_state = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
        base_obs = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]
        state_of = fill(base_state, ntrials)
        obs_of = fill(base_obs, ntrials)
    else
        cell_slds = _slds_cell_sldss(slds, grp)
        cell_state = [
            [_extract_state_params(lds.state_model) for lds in sc.LDSs] for sc in cell_slds
        ]
        cell_obs = [
            [_extract_obs_params(lds.obs_model) for lds in sc.LDSs] for sc in cell_slds
        ]
        state_of = [cell_state[grp.trial_cell[n]] for n in 1:ntrials]
        obs_of = [cell_obs[grp.trial_cell[n]] for n in 1:ntrials]
    end

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
            state_of[trial],
            obs_of[trial],
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
                    obs_params[k1].D * uy_trial[:, 1],
                ),
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
                        obs_params[k_curr].D * uy_trial[:, t],
                    ),
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

"""
    smooth(slds, y, w; ux=nothing, uy=nothing)

Smooth a single trial under **given** discrete responsibilities `w` (`K × T`), without
inferring them. Returns `(x_smooth, p_smooth)`. To infer the responsibilities as well,
call [`smooth(slds, y)`](@ref) with no `w`.
"""
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
    smooth(slds, y; ux=nothing, uy=nothing, smoothing_iters=100, tol=1e-6,
           return_cov=false, progress=false, depends_on=nothing)

Infer the joint posterior of a **fitted** `SLDS` with the parameters held fixed: the
continuous states `q(x)`, the discrete responsibilities `γₜ(k) = q(zₜ = k)`, and the
ELBO at those posteriors.

Alternates forward-backward over the switching chain with the Laplace/Kalman smoother
over the continuous states (Ghahramani & Hinton, 1996). Unlike the single-sample E-step
`fit!` runs during learning, the coupling here is deterministic — the discrete layer is
scored at the smoothed posterior mean, not at a draw from `q(x)` — so the result is
reproducible.

That plug-in is what makes `γ` reproducible, and it is what limits it: the mean is a
shrunk version of the latent path, so regimes that differ *only* in their dynamics
(`Aₖ`/`Qₖ`, with the emission shared) become hard to separate once the observation noise
is large relative to the process noise. Regimes that differ in their emissions, or whose
latent path the data pins down, are recovered sharply.

`fit!` runs the same alternation but keeps its forward-backward storage private, so this
is the way to read `q(z)` — a regime occupancy over time, a Viterbi-style `argmax` path,
a rate averaged over regimes — out of a model, on the data it was fitted to or on
held-out data.

`y` takes the same three shapes as [`fit!`](@ref), and `ux` / `uy` the same shape family.

# Keywords
- `smoothing_iters::Int=100`: maximum discrete↔continuous alternations.
- `tol::Real=1e-6`: stop once `max|Δγ| < tol`; `tol=0` runs exactly `smoothing_iters`
  alternations with no stopping test.
- `return_cov::Bool=false`: also return the smoothed covariances (`latent_dim² × T` per
  trial — large, hence opt-in).
- `progress::Bool=false`: show a progress bar.
- `depends_on`: optional `NamedTuple` of per-trial label vectors overriding the
  `depends_on` declared on the regimes for this call. A held-out set has its own trial
  count, so it needs its own label vectors.

# Returns
A `NamedTuple` `(; x, γ, elbo, p)`. For a single-trial matrix `y`, `x` is
`latent_dim × T`, `γ` is `K × T`, and `p` is `latent_dim × latent_dim × T`; for a 3-D
array or a vector of matrices, each is a `Vector` with one entry per trial. `elbo` is
always a scalar, and `p` is `nothing` unless `return_cov=true`.

Because a converged alternation is expensive, `smooth` returns everything it computed in
one call — read its `elbo` field rather than calling [`elbo`](@ref) separately.
"""
function smooth(
    slds::SLDS{T,S,O},
    y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}};
    ux=nothing,
    uy=nothing,
    smoothing_iters::Int=100,
    tol::Real=1e-6,
    return_cov::Bool=false,
    progress::Bool=false,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    #=
    Same setup as `fit!`, minus the M-step workspaces: `Data` validates and
    canonicalizes the observation / input shapes, the grouping resolves
    `depends_on` into per-cell parameter views, and the discrete layer wraps the
    (K × ΣT) log-likelihood matrix the forward-backward pass reads.
    =#
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    y_seq = data.y
    ux_seq = data.ux
    uy_seq = data.uy

    K = length(slds.LDSs)
    tsteps_per_trial = data.tsteps
    ntrials = length(y_seq)
    seq_ends = cumsum(tsteps_per_trial)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_per_trial)

    grp = _slds_parameter_grouping(slds, ntrials; depends_on=depends_on, y=y_seq)
    cell_slds = grp === nothing ? nothing : _slds_cell_sldss(slds, grp)

    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_per_trial)::TrialFilterSmooth{T}
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))
    fb_storage = _make_slds_fb_storage(dl, seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)
    slds_ws = SLDSSmoothWorkspace(T, slds, T_max)
    cell_ws = _slds_cell_workspaces(slds, cell_slds, slds_ws, T_max)

    #=
    No M-step runs, so the per-regime constants cached by the workspace stay
    valid for every alternation. `x_samples === nothing` throughout: the discrete
    layer is scored at the smoothed mean, so no draw from q(x) is ever needed.
    =#
    _slds_warmstart!(
        slds,
        cell_slds,
        grp,
        tfs,
        y_seq,
        nothing,
        slds_ws,
        tsteps_per_trial,
        K;
        ux=ux_seq,
        uy=uy_seq,
        cell_ws=cell_ws,
    )

    prog = if progress
        Progress(smoothing_iters; desc="Smoothing SLDS...", barlen=50, showspeed=true)
    else
        nothing
    end

    _, converged = _vem_alternate!(
        slds,
        cell_slds,
        grp,
        tfs,
        fb_storage,
        dl,
        y_seq,
        slds_ws;
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux_seq,
        uy=uy_seq,
        cell_ws=cell_ws,
        smoothing_iters=smoothing_iters,
        tol=T(tol),
        prog=prog,
    )
    prog !== nothing && finish!(prog)

    if tol > 0 && !converged
        @warn "SLDS smoothing did not converge" smoothing_iters tol
    end

    total_elbo = if grp === nothing
        elbo!(slds, tfs, fb_storage, y_seq, slds_ws; seq_ends=seq_ends, ux=ux_seq, uy=uy_seq)
    else
        _elbo_grouped!(
            cell_slds::Vector,
            grp::ParameterGrouping,
            tfs,
            fb_storage,
            y_seq,
            slds_ws;
            seq_ends=seq_ends,
            ux=ux_seq,
            uy=uy_seq,
            cell_ws=cell_ws,
        )
    end

    γ_trials = Vector{Matrix{T}}(undef, ntrials)
    x_trials = Vector{Matrix{T}}(undef, ntrials)
    p_trials = return_cov ? Vector{Array{T,3}}(undef, ntrials) : nothing
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        γ_trials[trial] = Matrix{T}(view(fb_storage.γ, :, t1:t2))
        x_trials[trial] = copy(tfs[trial].x_smooth)
        return_cov && (p_trials[trial] = copy(tfs[trial].p_smooth))
    end

    return _collect_slds_smooth_output(x_trials, γ_trials, p_trials, total_elbo, y)
end

#=
Public-shape return convention, mirroring `_collect_smooth_output` in fit_LDS.jl:
matrix in → per-trial arrays out (single trial); vector / 3-D array in → vectors out.
`y` is only inspected for its container type.
=#
function _collect_slds_smooth_output(x, γ, p, total_elbo, ::AbstractMatrix)
    return (; x=x[1], γ=γ[1], elbo=total_elbo, p=(p === nothing ? nothing : p[1]))
end

function _collect_slds_smooth_output(x, γ, p, total_elbo, _)
    return (; x=x, γ=γ, elbo=total_elbo, p=p)
end

"""
    _slds_fill_logL!(slds, cell_slds, grp, dl, y, x_of, slds_ws; seq_ends, ux, uy, cell_ws)

Fill `dl.logL` (`K × sum(T_i)`) with every regime's log-density of the current
continuous trajectory. `x_of(trial)` supplies that trajectory: the smoothed mean
for deterministic inference, a joint draw from `q(x)` for the Monte-Carlo E-step.

When `grp === nothing` this is the plain per-trial loop; otherwise trials are
visited cell by cell so the regime constants are refreshed once per cell.
"""
function _slds_fill_logL!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_of,
    slds_ws::SLDSSmoothWorkspace{T};
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    cell_ws::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    K = length(slds.LDSs)

    function fill_trial!(slds_t, ws_t, trial)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        x_src = x_of(trial)
        y_trial = y[trial]
        ux_trial = ux === nothing ? nothing : ux[trial]
        uy_trial = uy === nothing ? nothing : uy[trial]
        for k in 1:K
            joint_loglikelihood!(
                view(dl.logL, k, t1:t2),
                ws_t,
                ws_t.consts[k],
                slds_t.LDSs[k],
                x_src,
                y_trial,
                ux_trial,
                uy_trial,
            )
        end
        return nothing
    end

    if grp === nothing || cell_slds === nothing
        for trial in eachindex(y)
            fill_trial!(slds, slds_ws, trial)
        end
        return nothing
    end

    for c in 1:(grp.ncells)
        slds_c = cell_slds[c]
        ws_c = _slds_ws_for(cell_ws, slds_ws, c)
        refresh_slds_constants!(ws_c, slds_c)
        for trial in grp.cell_trials[c]
            fill_trial!(slds_c, ws_c, trial)
        end
    end
    return nothing
end

"""
    _slds_smooth_all!(slds, cell_slds, grp, tfs, y, x_samples, slds_ws, w_of; ...)

Run the Laplace/Newton smoother over every trial under the discrete weights
`w_of(trial)` (`K × T_i`), filling `tfs[*].x_smooth`, `tfs[*].p_smooth`, and
`tfs[*].entropy`. `x_samples === nothing` skips the joint draw (the
deterministic path); otherwise the next draw from `q(x)` lands in
`x_samples[trial]`.

When `grp === nothing` this is the plain per-trial loop; otherwise trials are
visited cell by cell so the regime constants are refreshed once per cell.
"""
function _slds_smooth_all!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    slds_ws::SLDSSmoothWorkspace{T},
    w_of;
    rng::AbstractRNG=Random.default_rng(),
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    cell_ws::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    if grp === nothing || cell_slds === nothing
        for trial in eachindex(y)
            smooth!(
                slds,
                tfs[trial],
                y[trial],
                w_of(trial);
                ws=slds_ws,
                x_sample=(x_samples === nothing ? nothing : x_samples[trial]),
                rng=rng,
                ux=(ux === nothing ? nothing : ux[trial]),
                uy=(uy === nothing ? nothing : uy[trial]),
            )
        end
        return nothing
    end

    for c in 1:(grp.ncells)
        _slds_smooth_cell!(
            cell_slds,
            grp,
            c,
            tfs,
            y,
            x_samples,
            slds_ws,
            w_of;
            rng=rng,
            ux=ux,
            uy=uy,
            cell_ws=cell_ws,
        )
    end
    return nothing
end

"""
    _vem_alternate!(slds, cell_slds, grp, tfs, fb_storage, dl, y, slds_ws; smoothing_iters, tol, x_samples, ...)

Run up to `smoothing_iters` discrete↔continuous alternations of the structured-
variational E-step. One alternation is:

1. fill `dl.logL` (`K × sum(T_i)`) with per-regime log-likelihoods of the current
   continuous trajectory,
2. refresh `q(z) = γ` by forward-backward over the switching chain (HMMs.jl threads
   across trials), and
3. refresh `q(x)` by re-running the Laplace/Newton smoother on each trial under the new
   `γ`, filling `tfs[*].x_smooth`, `tfs[*].p_smooth`, and `tfs[*].entropy`.

`x_samples` selects how step 1 reads the continuous trajectory, and is the only
difference between the two callers:

- `x_samples === nothing` — plug in the smoothed mean `E_q[x]`. Deterministic and
  reproducible; used by [`smooth`](@ref) for post-fit inference.
- `x_samples !== nothing` — plug in a joint draw from `q(x)`, and draw the next one in
  step 3. This is the vLEM Monte-Carlo E-step used by [`fit!`](@ref); `x_samples` is
  read then overwritten within each alternation.

`grp` / `cell_slds` carry an ancillary-dependency (`depends_on`) grouping; both
`nothing` keeps every step on the ungrouped code path.

`tol` selects the stopping rule. `tol == 0` runs exactly `smoothing_iters`
alternations; `tol > 0` stops early once `max|Δγ| < tol`.

Returns `(iters, converged)`.
"""
function _vem_alternate!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    cell_ws::Union{Nothing,AbstractVector}=nothing,
    smoothing_iters::Int,
    tol::T=zero(T),
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    prog=nothing,
) where {T<:Real}
    smoothing_iters >= 1 ||
        throw(ArgumentError("smoothing_iters must be ≥ 1, got $smoothing_iters"))

    K = length(slds.LDSs)

    # Deterministic path scores the smoothed mean; sampled path a draw from q(x).
    function x_of(trial)
        return x_samples === nothing ? tfs[trial].x_smooth : x_samples[trial]
    end

    function w_of(trial)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        return view(fb_storage.γ, :, t1:t2)
    end

    # Previous-iteration γ snapshot; only allocated when there is a stopping test.
    γ_prev = tol > 0 ? fill(T(Inf), K, last(seq_ends)) : nothing
    converged = false
    iters = 0

    for iter in 1:smoothing_iters
        iters = iter

        # (1) Score the current continuous trajectory under each regime.
        _slds_fill_logL!(
            slds,
            cell_slds,
            grp,
            dl,
            y,
            x_of,
            slds_ws;
            seq_ends=seq_ends,
            ux=ux,
            uy=uy,
            cell_ws=cell_ws,
        )

        # (2) Update q(z): single batched forward-backward across all trials.
        HMMs.forward_backward!(
            fb_storage,
            dl,
            obs_seq,
            control_seq;
            seq_ends=seq_ends,
            transition_marginals=true,
        )

        #=
        (3) Update q(x) under the fresh γ, drawing the next sample on the way
        out. Overwriting `x_samples` here is fine — step (1) already used the
        previous draw.
        =#
        _slds_smooth_all!(
            slds,
            cell_slds,
            grp,
            tfs,
            y,
            x_samples,
            slds_ws,
            w_of;
            rng=rng,
            ux=ux,
            uy=uy,
            cell_ws=cell_ws,
        )

        prog !== nothing && next!(prog)

        if γ_prev !== nothing
            if iter > 1
                Δγ = zero(T)
                @inbounds for i in eachindex(fb_storage.γ)
                    d = abs(fb_storage.γ[i] - γ_prev[i])
                    d > Δγ && (Δγ = d)
                end
                if Δγ < tol
                    converged = true
                    break
                end
            end
            copyto!(γ_prev, fb_storage.γ)
        end
    end

    return iters, converged
end

"""
    estep!(slds, tfs, fb_storage, dl, y, x_samples, slds_ws; rng, obs_seq, control_seq, seq_ends, smoothing_iters=1)

Monte-Carlo E-step for the SLDS: `smoothing_iters` coordinate-ascent alternations of
[`_vem_alternate!`](@ref), each scoring the discrete layer against a joint draw from
`q(x)` and drawing the next one. `smoothing_iters = 1` is the standard vLEM E-step;
larger values hand the M-step a better-converged posterior at proportional cost.

`x_samples` is read (to fill `dl.logL`) then overwritten (with the fresh draw) within
each alternation. `obs_seq`/`control_seq` are the HMMs.jl placeholder sequences built in
`fit!` (timestep indices / `nothing`s) — unrelated to the LDS control-input kwargs
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
    smoothing_iters::Int=1,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    _vem_alternate!(
        slds,
        nothing,
        nothing,
        tfs,
        fb_storage,
        dl,
        y,
        slds_ws;
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux,
        uy=uy,
        smoothing_iters=smoothing_iters,
        x_samples=x_samples,
        rng=rng,
    )
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
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    Tsteps = t2 - t1 + 1
    w = view(fb_storage.γ, :, t1:t2)  # K × Tsteps

    trial_elbo = zero(T)
    x_smooth_trial = fs.x_smooth

    #=
    Per-regime log-density scratch, built once rather than per regime. The
    `::AbstractVector{T}` is what keeps JET quiet: analysed at the unspecialized
    signature, `T` is only known to be `<:Real`, so `view` over the workspace
    field widens to a union carrying an `Any`-eltype branch, and no
    `joint_loglikelihood!` method matches that. Asserting the element type drops
    the branch; it always held.
    =#
    ll = view(slds_ws.ll_tmp, 1:Tsteps)::AbstractVector{T}

    # E_q[log p(y, x | z)], plug-in at the posterior mean, weighted by γ.
    for k in 1:K
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
    elbo(slds, y; ux=nothing, uy=nothing, smoothing_iters=100, tol=1e-6,
         progress=false, depends_on=nothing)

Evidence lower bound of an `SLDS` at the current parameters — the `elbo` field of
[`smooth`](@ref)`(slds, y)`, which infers `q(x)` and `q(z)` by deterministic
coordinate ascent before evaluating the bound. Deterministic and reproducible.

Accepts the same observation and input forms as [`smooth`](@ref), and the same
`smoothing_iters` / `tol` controls over the alternation. Returns a scalar.

If you also want the posteriors that produced it, call [`smooth`](@ref) once and read
its `elbo` field rather than paying for the alternation twice.
"""
function elbo(
    slds::SLDS{T,S,O},
    y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}};
    ux=nothing,
    uy=nothing,
    smoothing_iters::Int=100,
    tol::Real=1e-6,
    progress::Bool=false,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    return smooth(
        slds,
        y;
        ux=ux,
        uy=uy,
        smoothing_iters=smoothing_iters,
        tol=tol,
        return_cov=false,
        progress=progress,
        depends_on=depends_on,
    ).elbo
end

"""
    loglikelihood(slds, y; kwargs...)

Variational lower bound on the marginal log-likelihood of an `SLDS`, i.e.
[`elbo`](@ref)`(slds, y)`.

The exact marginal `log p(y)` is intractable for a switching model — it requires
summing over all `K^T` discrete regime sequences — so this returns the ELBO instead.
Values are comparable across models fit to the same data, but are lower bounds, not
likelihoods. Accepts the same keywords as [`elbo`](@ref).
"""
function StatsAPI.loglikelihood(slds::SLDS, y; kwargs...)
    return elbo(slds, y; kwargs...)
end

"""
    _tie_slots(tied::Bool, n) -> Vector{Int}

Slot vector over `n` units for one parameter group: every unit on slot 1 when
the group is tied, one slot each when it is not. Feeding these to the grouped
updates (`_grouped_update_A_b!` and friends) is what makes "tied across
regimes" and "free per regime" the same code path.
"""
_tie_slots(tied::Bool, n::Int) = tied ? ones(Int, n) : collect(1:n)

"""
    _validate_tied_params(lds, tied, grouped)

Reject the `tied_params` combinations the M-step has no estimator for.

A partial tie — some but not all columns of `[A b B]` or `[C d D]` — is fitted
by residualizing the free columns out and solving the shared block by
generalized least squares (`_partial_tied_regression`). That needs the group to
*be* a least-squares regression fitted from one set of sufficient statistics per
regime, which rules out two cases:

- a **Poisson** emission, whose `[C d D]` has no sufficient-statistic form and
  is fitted by LBFGS; and
- a fit that also groups trials with `depends_on`, where a regime's regression
  is several versions across cells rather than one, and a partial tie would have
  to partition columns and cells at once.

Both are fine with the whole regression tied.
"""
function _validate_tied_params(
    lds::LinearDynamicalSystem, tied::AbstractVector{Symbol}, grouped::Bool
)
    isempty(tied) && return nothing
    D = lds.latent_dim
    dyn = _tied_dyn_cols(tied, D, lds.ux_dim)
    obs = _tied_obs_cols(tied, D, lds.uy_dim)
    partial_dyn = !isempty(dyn) && length(dyn) < D + 1 + lds.ux_dim
    partial_obs = !isempty(obs) && length(obs) < D + 1 + lds.uy_dim

    if partial_obs && lds.obs_model isa PoissonObservationModel
        throw(
            ArgumentError(
                "tied_params: a Poisson emission's `[C d D]` is fitted by LBFGS, not " *
                "from sufficient statistics, so there is no way to share part of it " *
                "across regimes. Tie $(_join_names(_group_members(lds.obs_model, :C))) " *
                "together, or none of them.",
            ),
        )
    end

    if grouped && (partial_dyn || partial_obs)
        which = partial_dyn ? "`[A b B]`" : "`[C d D]`"
        throw(
            ArgumentError(
                "tied_params: sharing part of $which across regimes is not supported " *
                "alongside `depends_on`, which already splits that regression into one " *
                "version per group of trials. Tie the whole regression, or drop " *
                "`depends_on`.",
            ),
        )
    end
    return nothing
end

"""
    _broadcast_tied_params!(slds, tied)

Copy `LDSs[1]`'s tied parameters into every other regime.

The updates fit a shared value on the first regime that uses it, and an `SLDS`'s
regimes hold separate arrays rather than aliasing one (unlike the `depends_on`
variants, which share by reference), so the fitted value has to be copied out.
Copies by *column* for the stacked regressions, so a partial tie moves only the
shared columns and leaves each regime's free ones alone. Honours `fit_bool`: a
frozen group is left exactly as the caller set it, per regime.
"""
function _broadcast_tied_params!(
    slds::SLDS{T}, tied::AbstractVector{Symbol}
) where {T<:Real}
    isempty(tied) && return nothing
    src = slds.LDSs[1]
    D, p = src.latent_dim, src.obs_dim
    dyn_cols = _tied_dyn_cols(tied, D, src.ux_dim)
    obs_cols = _tied_obs_cols(tied, D, src.uy_dim)

    W_src = Matrix{T}(undef, D, D + 1 + src.ux_dim)
    W_dst = similar(W_src)
    V_src = Matrix{T}(undef, p, D + 1 + src.uy_dim)
    V_dst = similar(V_src)
    isempty(dyn_cols) || _pack_dyn_W!(W_src, src)
    isempty(obs_cols) || _pack_obs_V!(V_src, src)

    for k in 2:length(slds.LDSs)
        dst = slds.LDSs[k]
        if !isempty(dyn_cols) && dst.fit_bool[_G_AB]
            _pack_dyn_W!(W_dst, dst)
            @views W_dst[:, dyn_cols] .= W_src[:, dyn_cols]
            _unpack_dyn_W!(dst, W_dst)
        end
        if :Q in tied && dst.fit_bool[_G_Q]
            copyto!(dst.state_model.Q, src.state_model.Q)
        end
        if !isempty(obs_cols) && dst.fit_bool[_G_CD]
            _pack_obs_V!(V_dst, dst)
            @views V_dst[:, obs_cols] .= V_src[:, obs_cols]
            _unpack_obs_V!(dst, V_dst)
        end
        if :R in tied && dst.fit_bool[_G_R]
            copyto!(dst.obs_model.R, src.obs_model.R)
        end
    end
    return nothing
end

"""
    _tied_poisson_emission!(slds, tfs, data, sws, weights, tied)

Poisson emission M-step for an `SLDS`. Non-conjugate, so it is one LBFGS solve
per distinct `[C d D]`: `K` of them at the regimes' own responsibilities, or a
single unit-weight one when `[C d D]` is tied whole — summing the per-regime weighted
objectives collapses to the unit-weight one, because the emission term does not
depend on `k` and `Σₖ γₖ(t) = 1`.
"""
function _tied_poisson_emission!(
    slds::SLDS{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws::SmoothWorkspace{T},
    weights_of,
    tie_emission::Bool,
) where {T<:Real}
    if tie_emission
        update_observation_model!(slds.LDSs[1], tfs, data.y, [sws], nothing; uy=data.uy)
        return nothing
    end
    for k in eachindex(slds.LDSs)
        update_observation_model!(
            slds.LDSs[k], tfs, data.y, [sws], weights_of(k); uy=data.uy
        )
    end
    return nothing
end

#=
Which stacked regression an update is for. The two differ only in which blocks
of the sufficient statistics, which noise covariance and which prior they read,
so the tie logic is written once and dispatched on these.
=#
struct _DynBlock end
struct _ObsBlock end

_block_stats(::_DynBlock, suf) = (suf.dyn_xx[].mat, suf.dyn_xy)
_block_stats(::_ObsBlock, suf) = (suf.obs_xx[].mat, suf.obs_xy)

_block_noise(::_DynBlock, lds) = lds.state_model.Q
_block_noise(::_ObsBlock, lds) = lds.obs_model.R

_block_prior(::_DynBlock, lds) = lds.state_model.AB_prior
_block_prior(::_ObsBlock, lds) = lds.obs_model.CD_prior

_block_group(::_DynBlock) = _G_AB
_block_group(::_ObsBlock) = _G_CD

_block_width(::_DynBlock, lds) = lds.latent_dim + 1 + lds.ux_dim
_block_width(::_ObsBlock, lds) = lds.latent_dim + 1 + lds.uy_dim

_block_write!(::_DynBlock, lds, W) = _unpack_dyn_W!(lds, W)
_block_write!(::_ObsBlock, lds, W) = _unpack_obs_V!(lds, W)

function _block_grouped_update!(::_DynBlock, ldss, sufs, slots, noise_slots, sws, bufs)
    return _grouped_update_A_b!(ldss, sufs, slots, noise_slots, sws, bufs)
end

function _block_grouped_update!(::_ObsBlock, ldss, sufs, slots, noise_slots, sws, bufs)
    return _grouped_update_C_d!(ldss, sufs, slots, noise_slots, sws, bufs)
end

"""
    _slds_update_regression!(block, slds, sufs, tied_cols, noise_slots, sws, bufs, K)
        -> Vector{Int}

Fit one stacked regression across the regimes and return its slot vector — which
regimes ended up sharing a value, for the covariance update that follows.

Free or shared whole, the fit is the grouped update the `depends_on` path uses,
which picks the pooled or the generalized-least-squares estimator from whether
the regimes also share the covariance. A partial tie goes to
[`_partial_tied_regression`](@ref), which writes every regime's stacked matrix
directly — so its slots are all distinct: the regimes agree on the tied columns
and differ everywhere else.
"""
function _slds_update_regression!(
    block,
    slds::SLDS{T},
    sufs::AbstractVector,
    tied_cols::AbstractVector{Int},
    noise_slots::AbstractVector{Int},
    sws::SmoothWorkspace{T},
    bufs::GroupedSufBuffers{T},
    K::Int,
) where {T<:Real}
    lds1 = slds.LDSs[1]

    if isempty(tied_cols) || length(tied_cols) == _block_width(block, lds1)
        slots = _tie_slots(!isempty(tied_cols), K)
        _block_grouped_update!(block, slds.LDSs, sufs, slots, noise_slots, sws, bufs)
        return slots
    end

    slots = collect(1:K)
    lds1.fit_bool[_block_group(block)] || return slots

    stats = [_block_stats(block, sufs[k]) for k in 1:K]
    Ws = _partial_tied_regression(
        [st[1] for st in stats],
        [st[2] for st in stats],
        [_block_noise(block, slds.LDSs[k]) for k in 1:K],
        [_block_prior(block, slds.LDSs[k]) for k in 1:K],
        tied_cols,
        "tied_params",
    )
    for k in 1:K
        slds.LDSs[k].fit_bool[_block_group(block)] &&
            _block_write!(block, slds.LDSs[k], Ws[k])
    end
    return slots
end

"""
    mstep!(slds, tfs, fb_storage, dl, y, sws; obs_seq, seq_ends, ux=nothing, uy=nothing,
           tied=Symbol[])

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

`tied` names the parameter groups (canonical `:A` / `:Q` / `:C` / `:R`, from
`tied_params`) that every regime shares. Each becomes a one-slot group over the
`K` regimes and goes through the same `_grouped_update_*!` helpers the
`depends_on` path uses, then `_broadcast_tied_params!` copies the fitted value
into the other regimes. `x0`/`P0` are tied unconditionally, below.
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
    tied::AbstractVector{Symbol}=Symbol[],
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

    function weights_of(k)
        return [
            begin
                t1, t2 = HMMs.seq_limits(seq_ends, trial)
                view(fb_storage.γ, k, t1:t2)
            end for trial in 1:ntrials
        ]
    end

    #=
    One γ-weighted sufficient statistic per regime, all built before any update
    runs: a tied group is fitted from several regimes at once, and `Q` / `R`
    read the regression that was just written, so the updates cannot be
    interleaved with the aggregation the way a fully per-regime M-step can.
    =#
    sufs = [_initialize_td_sufficient_statistics(T, slds.LDSs[1], data.tsteps) for _ in 1:K]
    for k in 1:K
        _aggregate_td_suff_stats_weighted!(
            sufs[k], tfs, slds.LDSs[k], data, weights_of(k), sws
        )
    end

    lds1 = slds.LDSs[1]
    D = lds1.latent_dim
    dyn_cols = _tied_dyn_cols(tied, D, lds1.ux_dim)
    obs_cols = _tied_obs_cols(tied, D, lds1.uy_dim)

    slots_q = _tie_slots(:Q in tied, K)
    slots_r = _tie_slots(:R in tied, K)
    bufs = GroupedSufBuffers(T, lds1, data.tsteps)

    #=
    `[A b B]` then `Q`, `[C d D]` then `R`: the covariance updates read the
    regression that was just written. The returned slots also tell those updates
    how many distinct regressions there are, so a partial tie counts as `K` of
    them — every regime's stacked matrix differs, in its free columns.
    =#
    slots_ab = _slds_update_regression!(
        _DynBlock(), slds, sufs, dyn_cols, slots_q, sws, bufs, K
    )
    _grouped_update_Q!(slds.LDSs, sufs, slots_q, slots_ab, sws)

    if lds1.obs_model isa GaussianObservationModel{T}
        slots_cd = _slds_update_regression!(
            _ObsBlock(), slds, sufs, obs_cols, slots_r, sws, bufs, K
        )
        _grouped_update_R!(slds.LDSs, sufs, slots_r, slots_cd, sws)
    elseif lds1.obs_model isa PoissonObservationModel{T}
        _tied_poisson_emission!(
            slds, tfs, data, sws, weights_of, length(obs_cols) == D + 1 + lds1.uy_dim
        )
    else
        throw(ArgumentError("Unsupported observation model $(typeof(lds1.obs_model))"))
    end

    _broadcast_tied_params!(slds, tied)

    #=
    x0/P0 are tied across modes. Since the smoother gives one q(x) per trial
    and Σₖ γₖ(t=1) = 1, the pooled unit-weight init stats are exactly the sum
    over modes of the per-mode init stats the aggregator already computed.
    =#
    D = slds.LDSs[1].latent_dim
    suf = sufs[1]
    init_xy = zeros(T, 1, D)
    init_yy = zeros(T, D, D)
    init_n = zero(T)
    for k in 1:K
        init_xy .+= sufs[k].init_xy
        init_yy .+= sufs[k].init_yy[]
        init_n += T(sufs[k].init_n)
    end
    copyto!(suf.init_xy, init_xy)
    suf.init_yy[] = init_yy
    suf.init_n = init_n
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
    fit!(slds::SLDS, y; ux=nothing, uy=nothing, max_iter=50, smoothing_iters=1, progress=true)

Fit SLDS using variational Laplace EM. Runs for exactly `max_iter` iterations
(no early-stopping criterion: the E-step's posterior sampling makes the ELBO
trace noisy across iterations, so a tolerance check on successive differences
would fire spuriously). Returns the per-iteration ELBO trace.

Each E-step runs `smoothing_iters` discrete↔continuous alternations before the
M-step. The default of 1 is the standard vLEM update; larger values hand the
M-step a better-converged posterior at proportional cost per iteration.

`y` is a single trial `(obs_dim × T)` matrix, a `(obs_dim, T, ntrials)` array,
or a vector of per-trial matrices (ragged `T_i` allowed). Internally a single
batched `HMMs.ForwardBackwardStorage` of length `sum(T_i)` is allocated, with
`seq_ends = cumsum(T_i)` to demarcate trials.

Optional control inputs `ux` / `uy` accept the same shape family as `y`
(`nothing` when the regimes carry no `B` / `D`). They are shared across regimes
— the active regime `zₜ` selects which per-regime `Bₖ` / `Dₖ` multiplies the
input — and are re-estimated per regime alongside the other parameters. The
input dimensions must match across regimes (enforced by `validate_SLDS`).

Pass `depends_on` (a `NamedTuple` of per-trial label vectors) to override the
`depends_on` declared on the regimes' sub-models for this call. Every regime
must declare the same labels — the grouping of trials is a property of the data
— and `x0`/`P0` stay tied across regimes as usual.

Pass `tied_params` — a `Symbol` or a collection of them — to share parameters
across regimes instead of fitting one per regime. Names are the literal
parameter names `depends_on` and `fit_bool` use, and each means itself: `:C` is
`C`, not `[C d D]`. `tied_params = (:C, :d, :D, :R)` is the usual setup for
neural data — the recording does not change when the dynamics do, so only
`[A b B Q]` and the discrete chain switch, and the emission's parameter count is
divided by `K`. `tied_params = (:A, :b, :B, :Q)` is the mirror image: one set of
dynamics, switching emissions.

`:x0` / `:P0` are accepted and ignored — an SLDS ties its initial state across
regimes unconditionally. Combined with `depends_on` the tie is *within* a group:
each session keeps its own version, shared by every regime. Tied parameters are
broadcast before the first E-step, so no regime ever infers `q(x)` / `q(z)`
through a value the model does not have, and a frozen group (`fit_bool`) is left
exactly as the caller set it.

`[A b B]` and `[C d D]` are each fitted as one regression, so how much of one
you tie decides the cost. Tying it together with its covariance is the ordinary
pooled M-step (`O(m³)`), and so is tying a covariance on its own. Tying a
regression while its covariance still switches makes the shared fit a
generalized least-squares problem coupling the output rows (`O((p·m)³)`), and
tying only *part* of one adds a Frisch–Waugh projection in front of that. Both
are exact; tie the covariance alongside its regression when you can.

A partial tie has no reduction for a Poisson `[C d D]`, which is fitted by
LBFGS rather than from sufficient statistics, or alongside `depends_on`, which
already splits the regression per group of trials — those throw rather than
guess.
"""
function fit!(
    slds::SLDS{T,S,O},
    y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}};
    ux=nothing,
    uy=nothing,
    max_iter::Int=50,
    smoothing_iters::Int=1,
    progress::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
    depends_on::Union{Nothing,NamedTuple}=nothing,
    tied_params=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    tied = _resolve_tied_params(
        slds.LDSs[1].state_model, slds.LDSs[1].obs_model, tied_params
    )
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

    #=
    Ancillary parameter dependencies. `grp === nothing` (no regime declares
    `depends_on`) keeps every step on its original code path.
    =#
    grp = _slds_parameter_grouping(slds, ntrials; depends_on=depends_on, y=y_seq)
    cell_slds = grp === nothing ? nothing : _slds_cell_sldss(slds, grp)
    _validate_tied_params(slds.LDSs[1], tied, grp !== nothing)

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
    #=
    Per-cell workspaces for a stitching fit; `nothing` when ungrouped, and the
    base workspace itself when every session has the same channel count.
    =#
    cell_ws = _slds_cell_workspaces(slds, cell_slds, slds_ws, T_max)
    #=
    The M-step's regression buffers are shaped by `obs_dim` too, so a stitching
    fit needs one per cell. `_cell_workspace` shares the block-tridiagonal and
    smoothed-covariance storage, and `nothing` here keeps the single-workspace
    path for every fit whose cells have the parent's width.
    =#
    cell_mstep_sws = _slds_cell_mstep_workspaces(slds, cell_slds, sws, T_max)
    x_samples = [Matrix{T}(undef, latent_dim, Ti) for Ti in tsteps_per_trial]

    #=
    Broadcast the tied groups before the first E-step rather than only after the
    first M-step, so no regime ever infers `q(x)` / `q(z)` through a parameter
    the model does not have. Regimes seeded from one warm start already agree,
    and this makes that a property of the fit instead of an accident of the
    caller.
    =#
    if cell_slds === nothing
        _broadcast_tied_params!(slds, tied)
    else
        for slds_c in cell_slds
            _broadcast_tied_params!(slds_c, tied)
        end
    end

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
    _slds_warmstart!(
        slds,
        cell_slds,
        grp,
        tfs,
        y_seq,
        x_samples,
        slds_ws,
        tsteps_per_trial,
        K;
        rng=rng,
        ux=ux_seq,
        uy=uy_seq,
        cell_ws=cell_ws,
    )

    for iter in 1:max_iter
        #=
        E-step: fill q(z) from the current samples, run forward-backward,
        re-smooth q(x), and draw the next samples for the following iteration.
        =#
        if grp === nothing
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
                smoothing_iters=smoothing_iters,
            )

            # Compute the ELBO at the current posteriors.
            elbos[iter] = elbo!(
                slds,
                tfs,
                fb_storage,
                y_seq,
                slds_ws;
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
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
                tied=tied,
            )
            refresh_slds_constants!(slds_ws, slds)
        else
            grouping = grp::ParameterGrouping
            cells = cell_slds::Vector
            _estep_grouped!(
                cells,
                grouping,
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
                cell_ws=cell_ws,
                smoothing_iters=smoothing_iters,
            )

            elbos[iter] = _elbo_grouped!(
                cells,
                grouping,
                tfs,
                fb_storage,
                y_seq,
                slds_ws;
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
                cell_ws=cell_ws,
            )

            _mstep_grouped!(
                cells,
                grouping,
                tfs,
                fb_storage,
                dl,
                data,
                sws;
                obs_seq=obs_seq,
                seq_ends=seq_ends,
                cell_sws=cell_mstep_sws,
                tied=tied,
            )
        end

        prog !== nothing && next!(prog)
    end

    if prog !== nothing
        finish!(prog)
    end
    return elbos
end

# ============================================================================
# Ancillary parameter dependencies (`depends_on`) for the SLDS.
#
# The trial partition is a property of the *dataset*, so it must be identical
# across regimes; only the parameter values differ per regime. Given that, each
# cell is governed by one ordinary `SLDS` whose sub-LDSs hold that cell's
# parameter arrays, and the existing smoother / weighted aggregator run on it
# unchanged. Regime constants are refreshed once per cell rather than once per
# trial, so the per-cell overhead is K Cholesky sets per pass.
# ============================================================================

"""
    _slds_warmstart!(slds, cell_slds, grp, tfs, y, x_samples, slds_ws, tsteps, K; ...)

Smooth every trial once with uniform discrete weights `γ ≡ 1/K`, so the first
discrete update has a continuous trajectory to score. `x_samples` receives the
first posterior draw the Monte-Carlo E-step consumes; pass `nothing` for the
deterministic path of [`smooth`](@ref), which scores the smoothed mean instead.
"""
function _slds_warmstart!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    slds_ws::SLDSSmoothWorkspace{T},
    tsteps::AbstractVector{Int},
    K::Int;
    rng::AbstractRNG=Random.default_rng(),
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    cell_ws::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    function w_of(trial)
        return fill(one(T) / K, K, tsteps[trial])
    end

    _slds_smooth_all!(
        slds,
        cell_slds,
        grp,
        tfs,
        y,
        x_samples,
        slds_ws,
        w_of;
        rng=rng,
        ux=ux,
        uy=uy,
        cell_ws=cell_ws,
    )
    return nothing
end

"""
    _slds_parameter_grouping(slds, ntrials; depends_on=nothing)

Trial partition for an `SLDS`, or `nothing` when no regime declares
`depends_on`. Throws when the regimes disagree about the partition.
"""
function _slds_parameter_grouping(
    slds::SLDS, ntrials::Int; depends_on::Union{Nothing,NamedTuple}=nothing, y=nothing
)
    grp = parameter_grouping(slds.LDSs[1], ntrials; depends_on=depends_on, y=y)
    grp === nothing && return nothing
    for k in 2:length(slds.LDSs)
        grp_k = parameter_grouping(slds.LDSs[k], ntrials; depends_on=depends_on, y=y)
        ok =
            grp_k !== nothing &&
            grp_k.nslots == grp.nslots &&
            grp_k.cell_state == grp.cell_state &&
            grp_k.cell_obs == grp.cell_obs &&
            grp_k.trial_cell == grp.trial_cell
        ok || throw(
            ArgumentError(
                "SLDS: every regime must declare the same `depends_on` labels; regime " *
                "$k disagrees with regime 1. The grouping of trials is a property of " *
                "the data and is shared across regimes — only the fitted parameter " *
                "values differ per regime.",
            ),
        )
    end
    return grp
end

"""
    _slds_cell_sldss(slds, grp) -> Vector{SLDS}

One `SLDS` view per cell, sharing `A` and `πₖ` by reference (so the discrete
M-step still updates the single shared chain) and holding each regime's
per-cell parameter arrays.
"""
function _slds_cell_sldss(
    slds::SLDS{T,S,O,TM,ISV}, grp::ParameterGrouping
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel,TM,ISV}
    K = length(slds.LDSs)
    return [
        SLDS{T,S,O,TM,ISV}(slds.A, slds.πₖ, [_cell_lds(slds.LDSs[k], grp, c) for k in 1:K])
        for c in 1:(grp.ncells)
    ]
end

"""
    _cell_slds_workspace(base, slds_c, tsteps) -> SLDSSmoothWorkspace

One cell's SLDS workspace for a stitching fit. Reuses `base`'s
block-tridiagonal storage and per-timestep log-density scratch — the O(D²·T)
and O(T) parts, neither of which depends on `obs_dim` — and allocates fresh
per-regime constants and Newton buffers at this cell's channel count.

Safe for the same reason the LDS side is: cells run one at a time, and a cell's
Hessian blocks are consumed before the next cell overwrites them.
"""
function _cell_slds_workspace(
    base::SLDSSmoothWorkspace{T}, slds_c::SLDS, tsteps::Int
) where {T<:Real}
    lds1 = slds_c.LDSs[1]
    latent_dim = lds1.latent_dim
    obs_dim = lds1.obs_dim
    K = length(slds_c.LDSs)
    ws = SLDSSmoothWorkspace{T}(
        base.btd,                                          # shared O(D²·T)
        [SmoothConstants(T, latent_dim, obs_dim) for _ in 1:K],
        NewtonBuffers(T, latent_dim, obs_dim, tsteps),
        base.ll_tmp,                                       # shared, length T_max
    )
    refresh_slds_constants!(ws, slds_c)
    return ws
end

"""
    _slds_cell_workspaces(slds, cell_slds, base, tsteps) -> Vector or nothing

One workspace per cell. When every cell has the parent's `obs_dim` — which
includes every SLDS fit that is not stitching sessions of differing width —
each entry is `base` itself, so nothing extra is allocated and the previous
code path is what runs.
"""
function _slds_cell_workspaces(
    slds::SLDS,
    cell_slds::Union{Nothing,AbstractVector},
    base::SLDSSmoothWorkspace{T},
    tsteps::Int,
) where {T<:Real}
    cell_slds === nothing && return nothing
    p0 = slds.LDSs[1].obs_dim
    all(sc -> sc.LDSs[1].obs_dim == p0, cell_slds) && return [base for _ in cell_slds]
    return [_cell_slds_workspace(base, sc, tsteps) for sc in cell_slds]
end

_slds_ws_for(::Nothing, base::SLDSSmoothWorkspace, ::Int) = base
_slds_ws_for(cell_ws::AbstractVector, ::SLDSSmoothWorkspace, c::Int) = cell_ws[c]

"""
    _slds_cell_mstep_workspaces(slds, cell_slds, base, tsteps) -> Vector or nothing

One M-step `SmoothWorkspace` per cell, or `nothing` when every cell has the
parent's `obs_dim`. The regression buffers that fit `[Cₖ dₖ Dₖ]` and the
residual scatter `R` accumulates into are both shaped by the cell's channel
count; everything expensive is shared with `base`.
"""
function _slds_cell_mstep_workspaces(
    slds::SLDS,
    cell_slds::Union{Nothing,AbstractVector},
    base::SmoothWorkspace{T},
    tsteps::Int,
) where {T<:Real}
    cell_slds === nothing && return nothing
    lds1 = slds.LDSs[1]
    p0 = lds1.obs_dim
    all(sc -> sc.LDSs[1].obs_dim == p0, cell_slds) && return nothing
    return [
        _cell_workspace(
            base,
            lds1.latent_dim,
            sc.LDSs[1].obs_dim,
            tsteps;
            ux_dim=lds1.ux_dim,
            uy_dim=lds1.uy_dim,
        ) for sc in cell_slds
    ]
end

"""
    _slds_smooth_cell!(cell_slds, grp, cell, tfs, y, x_samples, slds_ws, w_of; rng, ux, uy)

Smooth every trial of one cell after refreshing the workspace's regime constants
for that cell's parameters. `w_of(trial)` supplies the `K × T` responsibilities.
"""
function _slds_smooth_cell!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
    cell::Int,
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    slds_ws::SLDSSmoothWorkspace{T},
    w_of;
    rng::AbstractRNG=Random.default_rng(),
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    cell_ws::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    slds_c = cell_slds[cell]
    ws_c = _slds_ws_for(cell_ws, slds_ws, cell)
    refresh_slds_constants!(ws_c, slds_c)
    for trial in grp.cell_trials[cell]
        smooth!(
            slds_c,
            tfs[trial],
            y[trial],
            w_of(trial);
            ws=ws_c,
            x_sample=(x_samples === nothing ? nothing : x_samples[trial]),
            rng=rng,
            ux=(ux === nothing ? nothing : ux[trial]),
            uy=(uy === nothing ? nothing : uy[trial]),
        )
    end
    return nothing
end

"""
    _estep_grouped!(cell_slds, grp, tfs, fb_storage, dl, y, x_samples, slds_ws; ...)

Grouped SLDS E-step: `smoothing_iters` alternations of [`_vem_alternate!`](@ref)
with the cell views in play, so each pass over the trials refreshes the regime
constants once per cell. The forward-backward call stays global — the discrete
chain is shared by all trials.
"""
function _estep_grouped!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
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
    cell_ws::Union{Nothing,AbstractVector}=nothing,
    smoothing_iters::Int=1,
) where {T<:Real}
    #=
    Cell 1 stands in for the parent only where `_vem_alternate!` needs a regime
    count and the ungrouped fall-through; every parameter read goes through
    `cell_slds` because `grp` is non-`nothing`.
    =#
    _vem_alternate!(
        cell_slds[1],
        cell_slds,
        grp,
        tfs,
        fb_storage,
        dl,
        y,
        slds_ws;
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux,
        uy=uy,
        cell_ws=cell_ws,
        smoothing_iters=smoothing_iters,
        x_samples=x_samples,
        rng=rng,
    )
    return nothing
end

"""
    _grouped_slds_prior_logdensity(cell_slds, grp, T)

`log p(θ)` for a grouped SLDS: the per-regime terms of
[`_slds_prior_logdensity`](@ref), counted once per distinct parameter version
instead of once per regime.
"""
function _grouped_slds_prior_logdensity(
    cell_slds::AbstractVector, grp::ParameterGrouping, ::Type{T}
) where {T<:Real}
    K = length(cell_slds[1].LDSs)
    total = zero(T)
    for k in 1:K
        ldss = [cell_slds[c].LDSs[k] for c in 1:(grp.ncells)]
        total += _grouped_state_prior_logdensity(ldss, grp.cell_slot, T)
        if ldss[1].obs_model isa GaussianObservationModel
            total += _grouped_gaussian_obs_prior_logdensity(ldss, grp.cell_slot, T)
        else
            total += _grouped_poisson_obs_prior_logdensity(ldss, grp.cell_slot, T)
        end
    end
    return total
end

"""
    _elbo_grouped!(cell_slds, grp, tfs, fb_storage, y, slds_ws; seq_ends, ux, uy)

Grouped SLDS ELBO: each trial's contribution evaluated against its cell's
parameters, plus one prior term per distinct parameter version.
"""
function _elbo_grouped!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    cell_ws::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    total_elbo = zero(T)
    for c in 1:(grp.ncells)
        slds_c = cell_slds[c]
        ws_c = _slds_ws_for(cell_ws, slds_ws, c)
        refresh_slds_constants!(ws_c, slds_c)
        for trial in grp.cell_trials[c]
            t1, t2 = HMMs.seq_limits(seq_ends, trial)
            total_elbo += _slds_trial_elbo(
                slds_c,
                tfs[trial],
                fb_storage,
                y[trial],
                ws_c,
                t1,
                t2,
                (ux === nothing ? nothing : ux[trial]),
                (uy === nothing ? nothing : uy[trial]),
            )
        end
    end
    return total_elbo + _grouped_slds_prior_logdensity(cell_slds, grp, T)
end

"""
    _broadcast_initial_state!(cell_slds, K, do_x0, do_P0)

Copy regime 1's initial-state parameters into every other regime. `x0`/`P0` are
tied across regimes, and the grouped update writes only into regime 1's
variants, so this restores the tie — including before the `P0` update, whose
scatter reads each unit's own `x0`.
"""
function _broadcast_initial_state!(
    cell_slds::AbstractVector, K::Int, do_x0::Bool, do_P0::Bool
)
    (do_x0 || do_P0) || return nothing
    for slds_c in cell_slds
        src = slds_c.LDSs[1].state_model
        for k in 2:K
            dst = slds_c.LDSs[k].state_model
            do_x0 && copyto!(dst.x0, src.x0)
            do_P0 && copyto!(dst.P0, src.P0)
        end
    end
    return nothing
end

"""
    _mstep_grouped!(cell_slds, grp, tfs, fb_storage, dl, data, sws; obs_seq, seq_ends,
                    tied=Symbol[])

Grouped SLDS M-step.

The discrete layer and the trial partition are shared, so the work is one
γ-weighted sufficient statistic per (regime, cell). Every parameter update then
runs over that flat unit list, driven by a slot vector per group:

- a group that is neither grouped nor tied gets one slot per unit — the plain
  per-(regime, cell) fit;
- `depends_on` makes cells sharing a version share a slot *within* a regime;
- naming the group in `tied_params` drops the regime out of the slot, so the
  cells' versions are shared across regimes as well. A tied `[C d D]` is then a
  property of the cell alone: each session keeps its own emission, shared by
  every regime, which is the usual reading for neural data.

Units are laid out regime-major, so the first unit of any version belongs to
regime 1; the update writes there and the broadcasters restore the tie. `x0`/`P0`
are tied across regimes unconditionally.
"""
function _mstep_grouped!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    data::Data{T},
    sws::SmoothWorkspace{T};
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    cell_sws::Union{Nothing,AbstractVector}=nothing,
    tied::AbstractVector{Symbol}=Symbol[],
) where {T<:Real}
    K = length(cell_slds[1].LDSs)
    ncells = grp.ncells
    lds1 = cell_slds[1].LDSs[1]

    # Discrete-layer M-step (slds.A, slds.πₖ are updated in place via dl).
    StatsAPI.fit!(dl, fb_storage, obs_seq; seq_ends=seq_ends)

    cell_data = [_subset_data(data, grp.cell_trials[c]) for c in 1:ncells]
    cell_tfs = [TrialFilterSmooth([tfs[n] for n in grp.cell_trials[c]]) for c in 1:ncells]
    bufs = GroupedSufBuffers(T, lds1, data.tsteps)

    function γ_view(k, trial)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        return view(fb_storage.γ, k, t1:t2)
    end

    unit_lds = [cell_slds[c].LDSs[k] for k in 1:K for c in 1:ncells]
    #=
    Sized from the unit's own LDS rather than cell 1's: under stitching each
    cell contributes a different number of channels, so `obs_xy` / `obs_yy`
    differ per unit. Identical to `lds1` whenever the widths agree.
    =#
    unit_suf = [
        _initialize_td_sufficient_statistics(T, cell_slds[c].LDSs[k], cell_data[c].tsteps)
        for k in 1:K for c in 1:ncells
    ]

    for k in 1:K, c in 1:ncells
        u = (k - 1) * ncells + c
        _aggregate_td_suff_stats_weighted!(
            unit_suf[u],
            cell_tfs[c],
            unit_lds[u],
            cell_data[c],
            [γ_view(k, n) for n in grp.cell_trials[c]],
            _unit_ws(cell_sws, sws, c),
        )
    end

    #=
    A cell's workspace, indexed by flat unit: `repeat` tiles the per-cell vector
    once per regime, so unit `(k-1)·ncells + c` lands on cell `c`'s entry.
    =#
    unit_sws = cell_sws === nothing ? nothing : repeat(cell_sws, K)

    #=
    A stacked regression is either shared whole across regimes or not at all
    here: `_validate_tied_params` rejects a partial tie alongside `depends_on`,
    which already splits the regression into one version per group of trials.
    =#
    D = lds1.latent_dim
    tie_dyn = length(_tied_dyn_cols(tied, D, lds1.ux_dim)) == D + 1 + lds1.ux_dim
    tie_obs = length(_tied_obs_cols(tied, D, lds1.uy_dim)) == D + 1 + lds1.uy_dim

    slots_ab = _grouped_unit_slots(grp.cell_slot[_G_AB], K, tie_dyn)
    slots_q = _grouped_unit_slots(grp.cell_slot[_G_Q], K, :Q in tied)
    slots_cd = _grouped_unit_slots(grp.cell_slot[_G_CD], K, tie_obs)
    slots_r = _grouped_unit_slots(grp.cell_slot[_G_R], K, :R in tied)

    _grouped_update_A_b!(unit_lds, unit_suf, slots_ab, slots_q, sws, bufs)
    _grouped_update_Q!(unit_lds, unit_suf, slots_q, slots_ab, sws)

    if lds1.obs_model isa GaussianObservationModel{T}
        _grouped_update_C_d!(
            unit_lds, unit_suf, slots_cd, slots_r, sws, bufs; unit_sws=unit_sws
        )
        _grouped_update_R!(unit_lds, unit_suf, slots_r, slots_cd, sws; unit_sws=unit_sws)
    elseif lds1.obs_model isa PoissonObservationModel{T}
        #=
        Non-conjugate: one LBFGS solve per `[C d D]` version, over the trials of
        every (regime, cell) unit sharing it. A version tied across regimes sees
        each of its trials once per regime, and `Σₖ γₖ(t) = 1`, so its weights
        collapse to the unit weights `nothing`.
        =#
        for units in _units_by_slot(slots_cd)
            trials = Int[]
            weights = Vector{SubArray{T,1}}()
            for u in units
                k, c = fldmod1(u, ncells)
                for n in grp.cell_trials[c]
                    push!(trials, n)
                    push!(weights, γ_view(k, n))
                end
            end
            order = sortperm(trials)
            unit_weights = _spans_all_regimes(units, ncells, K) ? nothing : weights[order]
            update_observation_model!(
                unit_lds[units[1]],
                TrialFilterSmooth([tfs[n] for n in trials[order]]),
                data.y[trials[order]],
                [_unit_ws(unit_sws, sws, units[1])],
                unit_weights;
                uy=data.uy[trials[order]],
            )
        end
    else
        throw(ArgumentError("Unsupported observation model $(typeof(lds1.obs_model))"))
    end

    #=
    Cells sharing a version share its arrays, so copying per cell is idempotent;
    doing it per cell rather than per version keeps this correct for any slot
    layout.
    =#
    for slds_c in cell_slds
        _broadcast_tied_params!(slds_c, tied)
    end

    #=
    Tied initial state, pooled over every (regime, cell) unit. Since
    Σₖ γₖ(t=1) = 1, summing the per-regime weighted init stats reproduces the
    unit-weight pooled statistic the ungrouped path uses.
    =#
    slots_x0 = repeat(grp.cell_slot[_G_X0], K)
    slots_P0 = repeat(grp.cell_slot[_G_P0], K)
    _grouped_update_x0!(unit_lds, unit_suf, slots_x0, bufs)
    _broadcast_initial_state!(cell_slds, K, lds1.fit_bool[_G_X0], false)
    _grouped_update_P0!(unit_lds, unit_suf, slots_P0, slots_x0, sws)
    _broadcast_initial_state!(cell_slds, K, false, lds1.fit_bool[_G_P0])

    return nothing
end

"""
    _grouped_unit_slots(cell_slot, K, tied) -> Vector{Int}

Slot vector over the `K · ncells` regime-major units for one parameter group,
from the group's per-cell slots.

Untied, a regime's cells get slots of their own, so no version is shared across
regimes. Tied, the cell's slot is used as-is and the same version spans every
regime — which is what makes a tied group a property of the cell rather than of
the (regime, cell) pair.
"""
function _grouped_unit_slots(cell_slot::AbstractVector{Int}, K::Int, tied::Bool)
    tied && return repeat(cell_slot, K)
    stride = maximum(cell_slot)
    return [(k - 1) * stride + s for k in 1:K for s in cell_slot]
end

"""
    _spans_all_regimes(units, ncells, K) -> Bool

Whether the flat regime-major `units` cover every regime for each cell they
touch — i.e. the version is tied across regimes, so `Σₖ γₖ(t) = 1` collapses its
responsibilities to unit weights.
"""
function _spans_all_regimes(units::AbstractVector{Int}, ncells::Int, K::Int)
    cells = unique(mod1.(units, ncells))
    return length(units) == K * length(cells)
end
