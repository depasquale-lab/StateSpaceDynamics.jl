#=============================================================================
Switching LDS (SLDS)

    Sample:         rand(rng, slds, tsteps)

    Log-Likelihood: joint_loglikelihood!(ws, slds, x, y, w)

    Gradient:       gradient!(ws, slds, x, y, w)

    Hessian:        hessian!(ws, slds, x, y, w)

    Smooth:         smooth!(slds, fs, y, w)

    Posterior:      sample_posterior!(x_out, rng, tfs, randn_buf)
                    sample_posterior(rng, fs)

    E-Step:         estep!(slds, tfs, fb_storage, dl, y, x_samples, slds_ws)

    M-Step:         mstep!(slds, tfs, fb_storage, dl, y, sws)

    Fit:            fit!(slds, y)
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
    obs_inputs = 1:total_T
    latent_inputs = fill(nothing, total_T)
    return HMMs.initialize_forward_backward(
        dl, obs_inputs, latent_inputs; seq_ends=seq_ends, transition_marginals=true
    )
end

"""
    rand([rng,] slds, tsteps::Integer)
    rand([rng,] slds, tsteps_per_trial::AbstractVector{<:Integer})

Sample from a Switching Linear Dynamical System.

- Scalar `tsteps`: returns one trial as `(z::Vector{Int}, x::Matrix, y::Matrix)`.
- Vector of per-trial lengths: returns `(z::Vector{Vector{Int}}, x::Vector{Matrix},
  y::Vector{Matrix})`. Trial lengths may differ.
"""
function Random.rand(
    rng::AbstractRNG, slds::SLDS{T,S,O}, tsteps::Integer
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    z = Vector{Int}(undef, Int(tsteps))
    x = Matrix{T}(undef, latent_dim, Int(tsteps))
    y = Matrix{T}(undef, obs_dim, Int(tsteps))

    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]

    _sample_slds_trial!(
        rng, z, x, y, slds.A, slds.πₖ, state_params, obs_params, slds.LDSs[1].obs_model
    )

    return z, x, y
end

function Random.rand(
    rng::AbstractRNG, slds::SLDS{T,S,O}, tsteps_per_trial::AbstractVector{<:Integer}
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    ntrials = length(tsteps_per_trial)

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
            slds.LDSs[1].obs_model,
        )
    end

    return z, x, y
end

function Random.rand(slds::SLDS, tsteps::Integer)
    return rand(Random.default_rng(), slds, tsteps)
end

function Random.rand(slds::SLDS, tsteps_per_trial::AbstractVector{<:Integer})
    return rand(Random.default_rng(), slds, tsteps_per_trial)
end

# Core SLDS trial sampling logic
function _sample_slds_trial!(
    rng, z_trial, x_trial, y_trial, A, πₖ, state_params, obs_params, obs_model_type
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
        rng, x_trial, y_trial, z_trial, state_params, obs_params, obs_model_type
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
)
    tsteps = length(z_trial)

    # Initial state from the selected LDS
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand(
        rng, MvNormal(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d, obs_params[k1].R)
    )

    # Subsequent states - switch dynamics based on discrete state
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t - 1], z_trial[t]

        # Continuous state follows previous discrete state's dynamics
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] + state_params[k_curr].b,
                state_params[k_curr].Q,
            ),
        )

        # Observation follows current discrete state's model
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d,
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
)
    tsteps = length(z_trial)

    # Initial state
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] =
        rand.(rng, Poisson.(exp.(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d)))

    # Subsequent states
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t - 1], z_trial[t]

        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] + state_params[k_curr].b,
                state_params[k_curr].Q,
            ),
        )

        y_trial[:, t] =
            rand.(
                rng,
                Poisson.(exp.(obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d)),
            )
    end
end

"""
    StatsAPI.fit!(dl::SLDSDiscreteLayer, fb_storage, obs_inputs; seq_ends)

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
    obs_inputs::AbstractVector;
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
    joint_loglikelihood!(ws, slds, x, y, w)

Compute weighted complete-data log-likelihood for SLDS.
Returns vector of per-timestep log-likelihoods.
"""
function joint_loglikelihood!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},   # K × T responsibilities/weights
) where {T<:Real}
    Tsteps = size(y, 2)

    # Workspace ll_vec may be sized for a longer trial; only touch the active prefix.
    ll_vec = ws.opt.ll_vec
    @views fill!(ll_vec[1:Tsteps], zero(T))

    K = length(slds.LDSs)
    for k in 1:K
        joint_loglikelihood!(
            view(ws.ll_tmp, 1:Tsteps), ws, ws.consts[k], slds.LDSs[k], x, y
        )
        for t in 1:Tsteps
            ll_vec[t] += w[k, t] * ws.ll_tmp[t]
        end
    end

    return view(ll_vec, 1:Tsteps)
end

"""
    gradient!(ws, slds, x, y, w)

In-place SLDS gradient: each component's complete-data gradient is scaled
per-timestep by the responsibility `w[k, t]` and accumulated. Writes into
`ws.opt.grad_buf` and returns it.
"""
function gradient!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},
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
        observation_gradient!(tmp1, cc, obs_buf, lds_k, x, y, 1)
        @. dxt = x[:, 1] - x0
        mul!(tmp3, neg_P0_inv, dxt)
        α = w[k, 1]
        @. grad[:, 1] += α * (tmp1 + tmp3)

        Tsteps == 1 && continue

        # Outgoing dynamics term comes from the factor at time 2, weighted by w[k,2]
        _transition_residual!(dxt_next, lds_k, x, 2)
        mul!(tmp2, A_inv_Q, dxt_next)
        @. grad[:, 1] += w[k, 2] * tmp2

        # 2 .. T-1: emission + incoming factor at t (w[k,t]),
        # outgoing factor at t+1 (w[k,t+1])
        for t in 2:(Tsteps - 1)
            observation_gradient!(tmp1, cc, obs_buf, lds_k, x, y, t)
            _transition_residual!(dxt, lds_k, x, t)
            mul!(tmp3, neg_Q_inv, dxt)
            α = w[k, t]
            @. grad[:, t] += α * (tmp1 + tmp3)

            _transition_residual!(dxt_next, lds_k, x, t + 1)
            mul!(tmp2, A_inv_Q, dxt_next)
            @. grad[:, t] += w[k, t + 1] * tmp2
        end

        # t = T: emission + incoming factor at T, weighted by w[k,T]
        observation_gradient!(tmp1, cc, obs_buf, lds_k, x, y, Tsteps)
        _transition_residual!(dxt, lds_k, x, Tsteps)
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
"""
function hessian!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},
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
            observation_hessian!(H_diag[1], cc, z, λ, lds_k, x, y, 1, α)
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
            observation_hessian!(H_diag[t], cc, z, λ, lds_k, x, y, t, w[k, t])
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
) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    tsteps = size(y, 2)
    n_active = latent_dim * tsteps

    ws === nothing && (ws = SLDSSmoothWorkspace(T, slds, tsteps))
    btd = ws.btd

    x = fs.x_smooth

    if all(fs.E_z .== 0)
        lds1 = slds.LDSs[1]
        x[:, 1] .= lds1.state_model.x0
        for t in 2:tsteps
            mul!(view(x, :, t), lds1.state_model.A, view(x, :, t - 1))
            x[:, t] .+= lds1.state_model.b
        end
    else
        copyto!(x, fs.E_z)
    end

    # Active-length views into (possibly) oversized workspace buffers.
    g = view(ws.opt.grad_buf, :, 1:tsteps)
    p = reshape(view(ws.opt.X0, 1:n_active), latent_dim, tsteps)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    ϕ!() = begin
        ll = joint_loglikelihood!(ws, slds, x, y, w)
        return sum(ll)
    end

    compute_grad! = (gcur, xcur) -> begin
        gradient!(ws, slds, xcur, y, w)
        copyto!(gcur, view(ws.opt.grad_buf, :, 1:tsteps))
        return nothing
    end

    build_hess! = (xcur) -> begin
        hessian!(ws, slds, xcur, y, w)
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
    hessian!(ws, slds, x, y, w)
    _negate_blocks!(btd, tsteps)

    block_tridiagonal_inverse!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    @views for t in 1:tsteps
        fs.p_smooth[:, :, t] .= Symmetrize!(fs.p_smooth[:, :, t])
    end

    return fs
end

# Public API wrapper
function smooth(slds::SLDS, y::AbstractMatrix{T}, w::AbstractMatrix{T}) where {T<:Real}
    fs = initialize_FilterSmooth(slds.LDSs[1], size(y, 2))::FilterSmooth{T}
    smooth!(slds, fs, y, w)
    return fs.x_smooth, fs.p_smooth
end

"""
    sample_posterior!(x_out, rng, tfs, randn_buf)

Sample one continuous-state trajectory per trial from the posterior, in place.
`x_out` is a `Vector{Matrix{T}}` with one matrix per trial sized at that trial's
length; trial lengths may differ. `randn_buf` (length `latent_dim`) is reused.
"""
function sample_posterior!(
    x_out::AbstractVector{<:AbstractMatrix{T}},
    rng::AbstractRNG,
    tfs::TrialFilterSmooth{T},
    randn_buf::Vector{T},
) where {T<:Real}
    min_jitter = T(1e-8)
    ntrials = length(tfs.FilterSmooths)

    for trial in 1:ntrials
        fs = tfs[trial]
        x_trial = x_out[trial]
        tsteps = size(x_trial, 2)
        for t in 1:tsteps
            chol = nothing
            jitter = zero(T)
            for attempt in 1:5
                try
                    chol = cholesky(Symmetric(fs.p_smooth[:, :, t]) + jitter * I)
                    break
                catch
                    jitter = min_jitter * T(10)^(attempt - 1)
                    if attempt == 5
                        @warn "Covariance not positive definite at t=$t trial=$trial, jitter=$jitter"
                        chol = cholesky(Symmetric(fs.p_smooth[:, :, t]) + jitter * I)
                    end
                end
            end
            randn!(rng, randn_buf)
            lmul!(chol.L, randn_buf)
            x_trial[:, t] .= fs.x_smooth[:, t] .+ randn_buf
        end
    end
    return x_out
end

"""
    sample_posterior(rng::AbstractRNG, fs::FilterSmooth{T}) where {T<:Real}

Sample a trajectory from the posterior over continuous states and compute its entropy.

Returns:
- x_sample: matrix of size (latent_dim, tsteps) representing one sample from
  q(x) = ∏ₜ N(x_t | x_smooth_t, p_smooth_t)
- entropy: H[q(x)] = ∑ₜ H[N(x_t | x_smooth_t, p_smooth_t)]
"""
function sample_posterior(rng::AbstractRNG, fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)
    x_sample = similar(fs.x_smooth)
    entropy = zero(T)
    min_jitter = T(1e-8)

    for t in 1:tsteps
        μ = fs.x_smooth[:, t]
        Σ = Symmetric(fs.p_smooth[:, :, t])

        # Try Cholesky decomposition with increasing jitter if needed
        chol = nothing
        jitter = zero(T)
        max_attempts = 5

        for attempt in 1:max_attempts
            try
                chol = cholesky(Σ + jitter * I)
                break
            catch
                if attempt == max_attempts
                    # Last resort: use larger jitter
                    jitter = min_jitter * T(10)^(attempt - 1)
                    @warn "Covariance matrix not positive definite at t=$t, adding jitter=$jitter"
                    chol = cholesky(Σ + jitter * I)
                else
                    # Increase jitter and try again
                    jitter = min_jitter * T(10)^(attempt - 1)
                end
            end
        end

        # Sample using the Cholesky factor
        Σ_chol = chol.L
        x_sample[:, t] = μ + Σ_chol * randn(rng, T, latent_dim)

        # Accumulate entropy using log determinant from Cholesky
        # log|Σ| = 2 * sum(log(diag(L))) where Σ = L*L'
        logdet_Σ = 2 * sum(log, diag(Σ_chol))
        entropy += 0.5 * (latent_dim * (1 + log(2π)) + logdet_Σ)
    end

    return x_sample, entropy
end

# Convenience method
function sample_posterior(fs::FilterSmooth{T}) where {T<:Real}
    return sample_posterior(Random.GLOBAL_RNG, fs)
end

"""
    estep!(slds, tfs, fb_storage, dl, y, x_samples, slds_ws; obs_inputs, latent_inputs, seq_ends)

E-step for SLDS using a single sample from the continuous posterior. Updates both
variational posteriors in coordinate-ascent order:

- Fills `dl.logL` (`K × sum(T_i)`) with per-state log-likelihoods from sampled continuous states
- Updates the discrete posterior q(z) via forward-backward (HiddenMarkovModels.jl, one
  storage covers all trials; HMMs.jl `@threads` across trials internally)
- Updates the continuous posterior q(x) by running the Laplace/Newton smoother on each
  trial with the freshly-updated discrete weights `γ`, filling `tfs[*].x_smooth`,
  `tfs[*].p_smooth`, and `tfs[*].entropy`.
"""
function estep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_samples::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    obs_inputs::AbstractVector,
    latent_inputs::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    ntrials = length(y)
    K = length(slds.LDSs)

    # Fill per-trial slices of dl.logL from the sampled continuous trajectory.
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        y_trial = y[trial]
        x_sample = x_samples[trial]
        for k in 1:K
            ll_view = view(dl.logL, k, t1:t2)
            joint_loglikelihood!(
                ll_view, slds_ws, slds_ws.consts[k], slds.LDSs[k], x_sample, y_trial
            )
        end
    end

    # Update q(z): single batched forward-backward (HMMs.jl threads across trials).
    HMMs.forward_backward!(
        fb_storage,
        dl,
        obs_inputs,
        latent_inputs;
        seq_ends=seq_ends,
        transition_marginals=true,
    )

    # Update q(x): smooth each trial's continuous states with the new discrete weights γ.
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        w = view(fb_storage.γ, :, t1:t2)  # K × Tsteps
        smooth!(slds, tfs[trial], y[trial], w; ws=slds_ws)
    end

    return nothing
end

"""
    elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends)
Compute the stochastic ELBO for SLDS at the current variational posteriors.
- Computes E_q[log p(y, x | z)] weighted by discrete posteriors
- Computes log p(z_1) and log p(z_t | z_{t-1}) weighted by discrete posteriors
- Subtracts entropies: -H[q(x)] - H[q(z)]
Returns a scalar ELBO value.

"""
function elbo!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    seq_ends::AbstractVector{Int},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    total_elbo = zero(T)
    ntrials = length(y)
    K = length(slds.LDSs)

    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        Tsteps = t2 - t1 + 1
        y_trial = y[trial]
        w = view(fb_storage.γ, :, t1:t2)  # K × Tsteps

        trial_elbo = zero(T)
        x_smooth_trial = tfs[trial].x_smooth

        # E_q[log p(y, x | z)] weighted by discrete posteriors.
        for k in 1:K
            ll = view(slds_ws.ll_tmp, 1:Tsteps)
            joint_loglikelihood!(
                ll, slds_ws, slds_ws.consts[k], slds.LDSs[k], x_smooth_trial, y_trial
            )
            for t in 1:Tsteps
                trial_elbo += w[k, t] * ll[t]
            end
        end

        # log p(z_1).
        for k in 1:K
            trial_elbo += w[k, 1] * log(slds.πₖ[k] + T(1e-12))
        end

        #=
        log p(z_t | z_{t-1}) = sum_t sum_{i,j} ξ[t][i,j] * log A[i,j].
        ξ is indexed by global timestep; the last entry of each trial (ξ[t2]) is zero
        by FB convention so we iterate t1..t2-1.
        =#
        for t in t1:(t2 - 1)
            ξt = fb_storage.ξ[t]
            for i in 1:K, j in 1:K
                trial_elbo += ξt[i, j] * log(slds.A[i, j] + T(1e-12))
            end
        end

        # Subtract entropies: -H[q(x)] - H[q(z)].
        trial_elbo -= tfs[trial].entropy
        for k in 1:K, t in 1:Tsteps
            wkt = w[k, t]
            wkt > 0 && (trial_elbo += wkt * log(wkt + T(1e-12)))
        end

        total_elbo += trial_elbo
    end

    return total_elbo
end

"""
    mstep!(slds, tfs, fb_storage, y, sws; obs_inputs, seq_ends)

M-step for SLDS.

- Updates discrete parameters (`slds.A`, `slds.πₖ`) via `StatsAPI.fit!` on the discrete
  layer (uses HMMs.jl's `ξ[t2]` scratch trick).
- Updates each LDS component using γ-weighted sufficient statistics aggregated
  by `_aggregate_td_suff_stats_weighted!`. For Gaussian sub-LDSs this is the
  full suf-based M-step (regression + IW MAP). For Poisson sub-LDSs the state-
  side updates flow through the same suf path; the emission [C d] is updated
  via the existing LBFGS routine (Poisson is non-conjugate and cannot be
  folded into the regression).
"""
function mstep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T};
    obs_inputs::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    ntrials = length(y)

    # Discrete-layer M-step (slds.A, slds.πₖ are updated in place via dl).
    StatsAPI.fit!(dl, fb_storage, obs_inputs; seq_ends=seq_ends)

    # SLDS doesn't currently expose user inputs; pass zero-column ux/uy.
    ux_seq = [zeros(T, 0, size(yt, 2)) for yt in y]
    uy_seq = [zeros(T, 0, size(yt, 2)) for yt in y]
    tsteps_per_trial = [size(yt, 2) for yt in y]

    # One reusable SufficientStatistics; overwritten per regime by the
    # weighted aggregator.
    suf = _initialize_td_sufficient_statistics(T, slds.LDSs[1], tsteps_per_trial)

    weights = Vector{AbstractVector{T}}(undef, ntrials)
    for k in 1:K
        lds_k = slds.LDSs[k]
        for trial in 1:ntrials
            t1, t2 = HMMs.seq_limits(seq_ends, trial)
            weights[trial] = view(fb_storage.γ, k, t1:t2)
        end

        _aggregate_td_suff_stats_weighted!(suf, tfs, lds_k, ux_seq, uy_seq, y, weights, sws)

        if lds_k.obs_model isa GaussianObservationModel{T}
            mstep!(lds_k, suf, sws)
        elseif lds_k.obs_model isa PoissonObservationModel{T}
            update_initial_state_mean!(lds_k, suf)
            update_initial_state_covariance!(lds_k, suf, sws)
            update_A_b!(lds_k, suf, sws)
            update_Q!(lds_k, suf, sws)
            #=
            SLDS owns a single sws (not a pool); wrap as a singleton so
            `update_observation_model!`'s threaded gradient path runs
            serially. SLDS Poisson is a niche path; the threading
            overhead isn't a meaningful win here.
            =#
            update_observation_model!(lds_k, tfs, y, [sws], weights)
        else
            throw(ArgumentError("Unsupported observation model $(typeof(lds_k.obs_model))"))
        end
    end

    return nothing
end

"""
    fit!(slds::SLDS, y::AbstractVector{<:AbstractMatrix}; max_iter=50, progress=true)
    fit!(slds::SLDS, y::AbstractMatrix; max_iter=50, progress=true)

Fit SLDS using variational Laplace EM with stochastic ELBO estimates. Runs for exactly
`max_iter` iterations.

`y` is either a single trial `(obs_dim × T)` matrix or a vector of per-trial matrices
(ragged `T_i` allowed). Internally a single batched `HMMs.ForwardBackwardStorage` of
length `sum(T_i)` is allocated, with `seq_ends = cumsum(T_i)` to demarcate trials.
"""
function fit!(
    slds::SLDS{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=50,
    progress::Bool=true,
    tol::Float64=1e-6,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    tsteps_per_trial = [size(yt, 2) for yt in y]
    ntrials = length(y)
    seq_ends = cumsum(tsteps_per_trial)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_per_trial)

    # Continuous-state smoother storage (per-trial sized).
    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_per_trial)::TrialFilterSmooth{T}

    # Discrete-layer wrapper (logL sized for batched obs_inputs).
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))

    # Single batched fb_storage covering all trials.
    fb_storage = _make_slds_fb_storage(dl, seq_ends)

    # Cached batched obs_inputs / latent_inputs for HMMs.jl.
    obs_inputs = collect(1:total_T)
    latent_inputs = fill(nothing, total_T)

    # Workspaces — allocated once at max trial length, reused each iteration.
    sws = SmoothWorkspace(T, latent_dim, obs_dim, T_max)
    slds_ws = SLDSSmoothWorkspace(T, slds, T_max)
    x_samples = [Matrix{T}(undef, latent_dim, Ti) for Ti in tsteps_per_trial]
    randn_buf = Vector{T}(undef, latent_dim)

    prog = if progress
        Progress(max_iter; desc="Fitting SLDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end
    elbos = Vector{T}(undef, max_iter)

    # Warm-start: smooth once per trial with uniform discrete weights.
    for trial in 1:ntrials
        Ti = tsteps_per_trial[trial]
        w_uniform = fill(one(T) / K, K, Ti)
        smooth!(slds, tfs[trial], y[trial], w_uniform; ws=slds_ws)
    end

    for iter in 1:max_iter

        # Sample one continuous-state trajectory per trial from the posterior.
        sample_posterior!(x_samples, Random.default_rng(), tfs, randn_buf)

        # E-step: compute discrete posteriors and smooth continuous states given sampled trajectories.
        estep!(
            slds,
            tfs,
            fb_storage,
            dl,
            y,
            x_samples,
            slds_ws;
            obs_inputs=obs_inputs,
            latent_inputs=latent_inputs,
            seq_ends=seq_ends,
        )

        # Compute stochastic ELBO estimate for this iteration.
        elbos[iter] = elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends)

        # M-step: update discrete and continuous parameters.
        mstep!(slds, tfs, fb_storage, dl, y, sws; obs_inputs=obs_inputs, seq_ends=seq_ends)
        refresh_slds_constants!(slds_ws, slds)

        prog !== nothing && next!(prog)
    end

    if prog !== nothing
        finish!(prog)
    end
    return elbos
end

function fit!(
    slds::SLDS{T,S,O}, y::AbstractMatrix{T}; kwargs...
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    return fit!(slds, [y]; kwargs...)
end
