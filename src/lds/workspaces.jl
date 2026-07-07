# =============================================================================
# Inference-state containers
#
# Per-trial smoothed estimates (`FilterSmooth` / `TrialFilterSmooth`) and the
# aggregated sufficient statistics (`SufficientStatistics`) that the E/M
# pipeline reads and writes. Defined here alongside the workspaces that house
# them. (`BlockTridiagonalWorkspace` lives in numerics/block_tridiagonal.jl,
# next to the solver that owns it.)
# =============================================================================

#=
Wrap a `PDMat` in a `Ref` whose element type matches the workspace field type
`PDMat{T,Matrix{T}}`. Pinning the eltype here keeps construction working on both 2- and 3-parameter
PDMats. Reading (`ref[]`) and rebinding (`ref[] = pd`) are unaffected either way. Not my favorite but it works.
TODO: Consider lowerbounding PDMats to 0.11.40
=#
_pd_ref(pd::PDMat{T,Matrix{T}}) where {T} = Base.RefValue{PDMat{T,Matrix{T}}}(pd)

"""
    FilterSmooth{T<:Real}

Per-trial container for smoothed estimates and associated covariance matrices.
A multi-trial fit holds one of these per trial (see `TrialFilterSmooth`); trial lengths
may differ.

# Fields
- `x_smooth::Matrix{T}`: smoothed state estimates `(latent_dim × T_trial)`
- `p_smooth::Array{T,3}`: smoothed covariances `(latent_dim × latent_dim × T_trial)`
- `p_smooth_tt1::Array{T,3}`: lag-1 cross covariances `(latent_dim × latent_dim × T_trial)`
- `E_z::Matrix{T}`: posterior mean `(latent_dim × T_trial)`
- `E_zz::Array{T,3}`: second moment `E[zₜzₜ']` `(latent_dim × latent_dim × T_trial)`
- `E_zz_prev::Array{T,3}`: second moment `E[zₜzₜ₋₁']` `(latent_dim × latent_dim × T_trial)`
- `entropy::T`: posterior entropy `H[q(x)]` for this trial
"""
mutable struct FilterSmooth{T<:Real}
    x_smooth::Matrix{T}
    p_smooth::Array{T,3}
    p_smooth_tt1::Array{T,3}
    E_z::Matrix{T}
    E_zz::Array{T,3}
    E_zz_prev::Array{T,3}
    entropy::T
end

function Base.show(io::IO, fs::FilterSmooth; gap="")
    println(io, gap, "Filter Smooth Object:")
    println(io, gap, "---------------------")
    println(io, gap, " size(x_smooth)  = ($(size(fs.x_smooth,1)), $(size(fs.x_smooth,2)))")
    println(
        io,
        gap,
        " size(p_smooth)  = ($(size(fs.p_smooth,1)), $(size(fs.p_smooth,2)), $(size(fs.p_smooth,3)))",
    )
    println(
        io,
        gap,
        " size(E_z)       = ($(size(fs.E_z,1)), $(size(fs.E_z,2)), $(size(fs.E_z,3)))",
    )
    println(
        io,
        gap,
        " size(E_zz)      = ($(size(fs.E_zz,1)), $(size(fs.E_zz,2)), $(size(fs.E_zz,3)), $(size(fs.E_zz,4)))",
    )
    println(
        io,
        gap,
        " size(E_zz_prev) = ($(size(fs.E_zz_prev,1)), $(size(fs.E_zz_prev,2)), $(size(fs.E_zz_prev,3)), $(size(fs.E_zz_prev,4)))",
    )

    return nothing
end

struct TrialFilterSmooth{T<:Real}
    FilterSmooths::Vector{FilterSmooth{T}}
end

Base.getindex(f::TrialFilterSmooth, i::Int) = f.FilterSmooths[i]
function Base.setindex!(
    f::TrialFilterSmooth, value::FilterSmooth{T}, i::Int
) where {T<:Real}
    return (f.FilterSmooths[i] = value)
end
Base.length(f::TrialFilterSmooth) = length(f.FilterSmooths)

mutable struct SufficientStatistics{T<:Real}

    # initial conditions. `init_n` is the effective sample count (e.g.
    # `ntrials` for unweighted fits; `Σₙ w[n,1]` for SLDS-style soft
    # responsibility weights). Stored as `T` rather than `Int` so the
    # weighted aggregator can flow non-integer counts through the M-step
    # without truncation.
    init_n::T
    init_xx::Base.RefValue{PDMat{T,Matrix{T}}}
    init_xy::Matrix{T}
    init_yy::Base.RefValue{PDMat{T,Matrix{T}}}

    # transitions model
    dyn_n::T
    dyn_xx::Base.RefValue{PDMat{T,Matrix{T}}}
    dyn_xy::Matrix{T}
    dyn_yy::Base.RefValue{PDMat{T,Matrix{T}}}

    # observation model
    obs_n::T
    obs_xx::Base.RefValue{PDMat{T,Matrix{T}}}
    obs_xy::Matrix{T}
    obs_yy::Base.RefValue{PDMat{T,Matrix{T}}}
end

"""
    SmoothWorkspace{T<:Real}

Pre-allocated workspace for the full LDS smoothing + EM pipeline.
Houses a `BlockTridiagonalWorkspace` for block tridiagonal operations, plus all
buffers needed by `loglikelihood!`, `Gradient!`, `Hessian!`, and M-step updates
to avoid repeated allocations during EM iterations.
"""
struct SmoothWorkspace{T<:Real}
    # Sub-workspace for block tridiagonal operations
    btd::BlockTridiagonalWorkspace{T}

    # Cached PDMats for R, Q, P0. Rewrapped once per E-step in
    # `compute_smooth_constants!`; downstream code consumes via
    # `ws.R_PD[].chol.U` (triangular factor) and `logdet(ws.R_PD[])`.
    # Mirrors `KalmanWorkspace`'s `Q_PD` / `P0_PD` / `R_PD` pattern.
    R_PD::Base.RefValue{PDMat{T,Matrix{T}}}      # (obs_dim × obs_dim)
    Q_PD::Base.RefValue{PDMat{T,Matrix{T}}}      # (latent_dim × latent_dim)
    P0_PD::Base.RefValue{PDMat{T,Matrix{T}}}     # (latent_dim × latent_dim)

    # Solve Outputs
    tmp_RC::Matrix{T}    # obs_dim × latent_dim   (R^{-1} C)
    tmp_QA::Matrix{T}    # latent_dim × latent_dim (Q^{-1} A)

    # Derived terms for Gradient
    C_inv_R::Matrix{T}        # (R_chol \ C)' = C'inv(R), size (latent_dim × obs_dim)
    A_inv_Q::Matrix{T}        # (Q_chol \ A)' = A'inv(Q), size (latent_dim × latent_dim)

    # Derived terms for Hessian block templates
    H_sub_entry::Matrix{T}    # Q_chol \ A, size (latent_dim × latent_dim)
    H_super_entry::Matrix{T}  # H_sub_entry', size (latent_dim × latent_dim)
    yt_given_xt::Matrix{T}    # -C'*(R_chol \ C), size (latent_dim × latent_dim)
    xt_given_xt_1::Matrix{T}  # -(Q_chol \ I), size (latent_dim × latent_dim)
    xt1_given_xt::Matrix{T}   # -A'*(Q_chol \ A), size (latent_dim × latent_dim)
    x_t::Matrix{T}            # -(P0_chol \ I), size (latent_dim × latent_dim)

    # Optimizer buffers (reused across EM iterations) 
    X₀::Vector{T}             # Vectorized latent path (latent_dim * tsteps)
    grad_buf::Matrix{T}       # Gradient output buffer (latent_dim × tsteps)
    grad_vec::Vector{T}       # Vectorized gradient for TwiceDifferentiable (latent_dim * tsteps)
    initial_h::SparseMatrixCSC{T,Int}  # Sparse Hessian (avoid copy each iteration)

    # Gradient temp vectors
    dxt::Vector{T}            # (latent_dim,)
    dxt_next::Vector{T}       # (latent_dim,)
    dyt::Vector{T}            # (obs_dim,)
    tmp1::Vector{T}           # (latent_dim,)
    tmp2::Vector{T}           # (latent_dim,)
    tmp3::Vector{T}           # (latent_dim,)

    # loglikelihood temp vectors
    ll_vec::Vector{T}         # (tsteps,)
    temp_dx::Vector{T}        # (latent_dim,)
    temp_dy::Vector{T}        # (obs_dim,)
    temp_solve_Q::Vector{T}   # (latent_dim,)
    temp_solve_R::Vector{T}   # (obs_dim,)

    # Hessian temp matrix
    I_mat::Matrix{T}          # Identity (latent_dim × latent_dim)

    # M-step buffers
    Sxz::Matrix{T}            # (latent_dim × latent_dim+1) for update_A_b!
    Szz_Ab::Matrix{T}         # (latent_dim+1 × latent_dim+1) for update_A_b!
    AB::Matrix{T}             # (latent_dim × latent_dim+1) for update_A_b!
    Syz::Matrix{T}            # (obs_dim × latent_dim+1) for update_C_d!
    Szz_Cd::Matrix{T}         # (latent_dim+1 × latent_dim+1) for update_C_d!
    Q_sum::Matrix{T}          # (latent_dim × latent_dim)
    R_sum::Matrix{T}          # (obs_dim × obs_dim)
    S0_sum::Matrix{T}         # (latent_dim × latent_dim)

    # update_C_d! temps
    CD::Matrix{T}             # (obs_dim × latent_dim + 1)

    # ELBO / Q_state buffers
    elbo_temp::Matrix{T}           # (latent_dim × latent_dim) - main accumulator
    elbo_sum_E_zz::Matrix{T}       # (latent_dim × latent_dim)
    elbo_sum_E_zzm1::Matrix{T}     # (latent_dim × latent_dim)
    elbo_sum_E_cross::Matrix{T}    # (latent_dim × latent_dim)
    elbo_sum_mu_t::Vector{T}       # (latent_dim,)
    elbo_sum_mu_tm1::Vector{T}     # (latent_dim,)
    elbo_temp2::Matrix{T}          # (latent_dim × latent_dim) - for A * sum_E_zzm1 * A'

    # Q_obs buffers (Gaussian)
    elbo_obs_temp::Matrix{T}       # (obs_dim × obs_dim) - accumulator
    elbo_obs_work::Matrix{T}       # (obs_dim × obs_dim) - work matrix
    elbo_ytil::Vector{T}           # (obs_dim,) - residualized y
    elbo_sum_yy::Matrix{T}         # (obs_dim × obs_dim)
    elbo_sum_yz::Matrix{T}         # (obs_dim × latent_dim)
    elbo_obs_work1::Matrix{T}      # (obs_dim × obs_dim)
    elbo_obs_work2::Matrix{T}      # (latent_dim × obs_dim)

    # Q_obs buffers (Poisson)
    h_obs::Vector{T}               # (obs_dim,) - h_t = C * E[x_t] + d
    rho_obs::Vector{T}             # (obs_dim,) - variance correction
    CP_obs::Matrix{T}              # (obs_dim × latent_dim) - C * P_t
    CEz_obs::Vector{T}             # (obs_dim,) - C * E[x_t]

    # Shared smoothed-covariance storage for the equal-length multi-trial fast
    # path. The BT Hessian (and therefore its inverse) is observation-
    # independent; when all trials of a fit share the same length, the
    # smoothed covariances `P_smooth[t]` and cross-covariances `P_smooth[t,t-1]`
    # are computed once on a designated workspace and aliased by every trial's
    # `FilterSmooth.p_smooth` / `p_smooth_tt1` field. Mirrors the Kalman
    # path's `p_smooth_shared` / `p_smooth_tt1_shared` pattern.
    p_smooth_shared::Array{T,3}      # (latent_dim, latent_dim, tsteps)
    p_smooth_tt1_shared::Array{T,3}  # (latent_dim, latent_dim, tsteps)

    # Aggregator output buffers, sized to match the shapes of
    # `SufficientStatistics`. The TD aggregator writes per-trial GEMM/SYRK
    # contributions into these, then wraps them as PDMats once per E-step.
    # Reused across iterations (only the contents change).
    td_init_xy::Matrix{T}              # (1, latent_dim)         Σₙ x_init
    td_dyn_xy::Matrix{T}               # (dyn_reg_dim, D)        Σₙ Σₜ [x_{t-1};1;u_{t-1}] xₜ'
    td_obs_xy::Matrix{T}               # (obs_reg_dim, p)        Σₙ Σₜ [xₜ;1;vₜ] yₜ'

    # Cross-trial smoothed-covariance accumulators
    td_sum_smooth_cov_prev::Matrix{T}  # (D, D)  Σₙ Σ_{t=1:Tₙ-1} P_smooth[t]
    td_sum_smooth_cov_next::Matrix{T}  # (D, D)  Σₙ Σ_{t=2:Tₙ}   P_smooth[t]
    td_sum_smooth_cov_all::Matrix{T}   # (D, D)  Σₙ Σ_{t=1:Tₙ}   P_smooth[t]
    td_sum_smooth_xcov::Matrix{T}      # (D, D)  Σₙ Σ_{t=2:Tₙ}   P_smooth_tt1[t]

    # Constant aggregates over the input data (filled once at fit entry, not
    # touched again). The y-only / uy-only blocks of obs_xx, obs_xy, obs_yy and
    # the ux-only blocks of dyn_xx are observation-independent so we cache them
    # here to skip re-summing every E-step.
    td_obs_yy_const::Matrix{T}         # (p, p)                Σₙ Σₜ yₜ yₜ'
    td_obs_xy_const::Matrix{T}         # (obs_reg_dim, p)      bias + uy-rows of obs_xy
    td_obs_xx_const::Matrix{T}         # (obs_reg_dim, obs_reg_dim) bias / uy blocks
    td_dyn_xx_const::Matrix{T}         # (dyn_reg_dim, dyn_reg_dim) bias / ux blocks

    # Batched mean-pass buffers (equal-length cov-cache fast path). Only the
    # designated `sws_pool[1]` workspace allocates these with `ntrials > 1`;
    # the rest of the pool keeps them at `ntrials = 1` (effectively empty).
    # The (D, T, N) tensors share storage with their `(D*T, N)` reshaped views
    # used as matrix RHS for `block_tridiagonal_backsubst!`.
    batched_x_mat::Array{T,3}         # (latent_dim, tsteps, ntrials) - current iterate
    batched_grad_buf::Array{T,3}      # (latent_dim, tsteps, ntrials) - Gradient! output
    batched_dxt::Matrix{T}            # (latent_dim, ntrials)
    batched_dxt_next::Matrix{T}       # (latent_dim, ntrials)
    batched_dyt::Matrix{T}            # (obs_dim, ntrials)
    batched_tmp1::Matrix{T}           # (latent_dim, ntrials)
    batched_tmp2::Matrix{T}           # (latent_dim, ntrials)
    batched_tmp3::Matrix{T}           # (latent_dim, ntrials)

    # Stacked observation / control tensors used by the batched mean pass.
    # Populated once at the first batched `smooth!` call (data is constant
    # across EM iters within a fit). 0-sized when `ntrials = 1`.
    batched_y::Array{T,3}             # (obs_dim, tsteps, ntrials)
    batched_ux::Array{T,3}             # (ux_dim, tsteps, ntrials)
    batched_uy::Array{T,3}             # (uy_dim, tsteps, ntrials)
    batched_data_valid::Base.RefValue{Bool}  # true after first populate
end

"""
    SmoothWorkspace(::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int;
                    ux_dim=0, uy_dim=0, ntrials=1)

Construct a preallocated `SmoothWorkspace` for the full LDS EM pipeline.

- `ux_dim` is the dynamics-input dimension (`size(state_model.B, 2)`), used to
  size the M-step regression buffers `Sxz`/`Szz_Ab`/`AB` to fit `[A b B]`.
- `uy_dim` is the observation-input dimension (`size(obs_model.D, 2)`), used
  to size `Syz`/`Szz_Cd`/`CD` to fit `[C d D]`.
- `ntrials` sizes the batched mean-pass buffers used by the equal-length
  cov-cache fast path. Default 1 (effectively empty). Only `sws_pool[1]` at
  fit entry needs the real `ntrials`; the rest of the pool keeps the default.

Either of `ux_dim` / `uy_dim` being zero (the default) means no inputs — buffers
fit `[A b]` and/or `[C d]` only.
"""
function SmoothWorkspace(
    ::Type{T},
    latent_dim::Int,
    obs_dim::Int,
    tsteps::Int;
    ux_dim::Int=0,
    uy_dim::Int=0,
    ntrials::Int=1,
) where {T<:Real}
    btd = BlockTridiagonalWorkspace(T, latent_dim, tsteps)

    # Placeholder PDMats — rewrapped at the start of every E-step
    # by `compute_smooth_constants!`.
    R_PD = _pd_ref(PDMat(Matrix{T}(I, obs_dim, obs_dim)))
    Q_PD = _pd_ref(PDMat(Matrix{T}(I, latent_dim, latent_dim)))
    P0_PD = _pd_ref(PDMat(Matrix{T}(I, latent_dim, latent_dim)))

    tmp_RC = zeros(T, obs_dim, latent_dim)
    tmp_QA = zeros(T, latent_dim, latent_dim)

    C_inv_R = zeros(T, latent_dim, obs_dim)
    A_inv_Q = zeros(T, latent_dim, latent_dim)
    H_sub_entry = zeros(T, latent_dim, latent_dim)
    H_super_entry = zeros(T, latent_dim, latent_dim)
    yt_given_xt = zeros(T, latent_dim, latent_dim)
    xt_given_xt_1 = zeros(T, latent_dim, latent_dim)
    xt1_given_xt = zeros(T, latent_dim, latent_dim)
    x_t = zeros(T, latent_dim, latent_dim)

    # Optimizer buffers
    X₀ = zeros(T, latent_dim * tsteps)
    grad_buf = zeros(T, latent_dim, tsteps)
    grad_vec = zeros(T, latent_dim * tsteps)
    initial_h = copy(btd.H_sparse)

    # Gradient temp vectors
    dxt = zeros(T, latent_dim)
    dxt_next = zeros(T, latent_dim)
    dyt = zeros(T, obs_dim)
    tmp1 = zeros(T, latent_dim)
    tmp2 = zeros(T, latent_dim)
    tmp3 = zeros(T, latent_dim)

    # loglikelihood temp vectors
    ll_vec = zeros(T, tsteps)
    temp_dx = zeros(T, latent_dim)
    temp_dy = zeros(T, obs_dim)
    temp_solve_Q = zeros(T, latent_dim)
    temp_solve_R = zeros(T, obs_dim)

    # Hessian temp matrix
    I_mat = Matrix{T}(I, latent_dim, latent_dim)

    # M-step buffers. The "+1" is for the affine bias column (b for the
    # dynamics regression, d for the observation regression); ux_dim / uy_dim
    # add the user input columns when controls are supplied.
    dyn_reg_dim = latent_dim + 1 + ux_dim
    obs_reg_dim = latent_dim + 1 + uy_dim
    Sxz = zeros(T, latent_dim, dyn_reg_dim)
    Szz_Ab = zeros(T, dyn_reg_dim, dyn_reg_dim)
    AB = zeros(T, latent_dim, dyn_reg_dim)
    Syz = zeros(T, obs_dim, obs_reg_dim)
    Szz_Cd = zeros(T, obs_reg_dim, obs_reg_dim)
    Q_sum = zeros(T, latent_dim, latent_dim)
    R_sum = zeros(T, obs_dim, obs_dim)
    S0_sum = zeros(T, latent_dim, latent_dim)
    CD = zeros(T, obs_dim, obs_reg_dim)

    # ELBO / Q_state buffers
    elbo_temp = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_zz = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_zzm1 = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_cross = zeros(T, latent_dim, latent_dim)
    elbo_sum_mu_t = zeros(T, latent_dim)
    elbo_sum_mu_tm1 = zeros(T, latent_dim)
    elbo_temp2 = zeros(T, latent_dim, latent_dim)

    # Q_obs buffers (Gaussian)
    elbo_obs_temp = zeros(T, obs_dim, obs_dim)
    elbo_obs_work = zeros(T, obs_dim, obs_dim)
    elbo_ytil = zeros(T, obs_dim)
    elbo_sum_yy = zeros(T, obs_dim, obs_dim)
    elbo_sum_yz = zeros(T, obs_dim, latent_dim)
    elbo_obs_work1 = zeros(T, obs_dim, obs_dim)
    elbo_obs_work2 = zeros(T, latent_dim, obs_dim)

    # Q_obs buffers (Poisson)
    h_obs = zeros(T, obs_dim)
    rho_obs = zeros(T, obs_dim)
    CP_obs = zeros(T, obs_dim, latent_dim)
    CEz_obs = zeros(T, obs_dim)

    # Shared smoothed-covariance storage for the equal-length multi-trial
    # fast path (filled once per E-step on the designated workspace, then
    # aliased by every trial's FilterSmooth).
    p_smooth_shared = zeros(T, latent_dim, latent_dim, tsteps)
    p_smooth_tt1_shared = zeros(T, latent_dim, latent_dim, tsteps)

    # Kalman-style aggregator buffers
    td_init_xy = zeros(T, 1, latent_dim)
    td_dyn_xy = zeros(T, dyn_reg_dim, latent_dim)
    td_obs_xy = zeros(T, obs_reg_dim, obs_dim)
    td_sum_smooth_cov_prev = zeros(T, latent_dim, latent_dim)
    td_sum_smooth_cov_next = zeros(T, latent_dim, latent_dim)
    td_sum_smooth_cov_all = zeros(T, latent_dim, latent_dim)
    td_sum_smooth_xcov = zeros(T, latent_dim, latent_dim)
    td_obs_yy_const = zeros(T, obs_dim, obs_dim)
    td_obs_xy_const = zeros(T, obs_reg_dim, obs_dim)
    td_obs_xx_const = zeros(T, obs_reg_dim, obs_reg_dim)
    td_dyn_xx_const = zeros(T, dyn_reg_dim, dyn_reg_dim)

    # Batched mean-pass buffers. Sized at `ntrials = 1` by default — only
    # `sws_pool[1]` at fit entry passes the actual ntrials so the batched
    # backsubst can do BLAS-3 across trials.
    batched_x_mat = zeros(T, latent_dim, tsteps, ntrials)
    batched_grad_buf = zeros(T, latent_dim, tsteps, ntrials)
    batched_dxt = zeros(T, latent_dim, ntrials)
    batched_dxt_next = zeros(T, latent_dim, ntrials)
    batched_dyt = zeros(T, obs_dim, ntrials)
    batched_tmp1 = zeros(T, latent_dim, ntrials)
    batched_tmp2 = zeros(T, latent_dim, ntrials)
    batched_tmp3 = zeros(T, latent_dim, ntrials)
    batched_y = zeros(T, obs_dim, tsteps, ntrials)
    batched_ux = zeros(T, ux_dim, tsteps, ntrials)
    batched_uy = zeros(T, uy_dim, tsteps, ntrials)
    batched_data_valid = Ref(false)

    return SmoothWorkspace{T}(
        btd,
        R_PD,
        Q_PD,
        P0_PD,
        tmp_RC,
        tmp_QA,
        C_inv_R,
        A_inv_Q,
        H_sub_entry,
        H_super_entry,
        yt_given_xt,
        xt_given_xt_1,
        xt1_given_xt,
        x_t,
        X₀,
        grad_buf,
        grad_vec,
        initial_h,
        dxt,
        dxt_next,
        dyt,
        tmp1,
        tmp2,
        tmp3,
        ll_vec,
        temp_dx,
        temp_dy,
        temp_solve_Q,
        temp_solve_R,
        I_mat,
        Sxz,
        Szz_Ab,
        AB,
        Syz,
        Szz_Cd,
        Q_sum,
        R_sum,
        S0_sum,
        CD,
        elbo_temp,
        elbo_sum_E_zz,
        elbo_sum_E_zzm1,
        elbo_sum_E_cross,
        elbo_sum_mu_t,
        elbo_sum_mu_tm1,
        elbo_temp2,
        elbo_obs_temp,
        elbo_obs_work,
        elbo_ytil,
        elbo_sum_yy,
        elbo_sum_yz,
        elbo_obs_work1,
        elbo_obs_work2,
        h_obs,
        rho_obs,
        CP_obs,
        CEz_obs,
        p_smooth_shared,
        p_smooth_tt1_shared,
        td_init_xy,
        td_dyn_xy,
        td_obs_xy,
        td_sum_smooth_cov_prev,
        td_sum_smooth_cov_next,
        td_sum_smooth_cov_all,
        td_sum_smooth_xcov,
        td_obs_yy_const,
        td_obs_xy_const,
        td_obs_xx_const,
        td_dyn_xx_const,
        batched_x_mat,
        batched_grad_buf,
        batched_dxt,
        batched_dxt_next,
        batched_dyt,
        batched_tmp1,
        batched_tmp2,
        batched_tmp3,
        batched_y,
        batched_ux,
        batched_uy,
        batched_data_valid,
    )
end

"""
    compute_smooth_constants!(ws::SmoothWorkspace{T}, lds)

Pre-compute and cache all Cholesky factors and derived terms that are constant
within a single `smooth!` call (i.e., depend only on model parameters, not on x).
Must be called once at the start of each `smooth!` invocation.

Dispatches on the observation model type:
- Gaussian: computes both state and observation model terms.
- Poisson: only computes state model terms (observation terms are x-dependent).
"""
function compute_smooth_constants!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R

    # Rewrap covariances as PDMats — each PDMat caches its own Cholesky
    # factor internally and is consumed downstream via `ws.X_PD[].chol.U`
    # for triangular solves and `logdet(ws.X_PD[])` for the normalizer.
    #
    # When `WT === T` (the hot path) `convert(Matrix{WT}, M)` returns `M`
    # unchanged — no copy, no alloc. When the workspace eltype differs
    # (e.g. `ForwardDiff.Dual` for autodiff `loglikelihood`), constructing
    # the PDMat directly with `WT`-typed factors avoids the
    # `convert(::Type{PDMat{WT}}, ::PDMat{T})` fallback that requires a
    # single-arg `Cholesky{WT}(::Cholesky{T})` method — present in
    # Julia 1.12 but not Julia 1.10's stdlib `LinearAlgebra`.
    R_w = convert(Matrix{WT}, R)
    Q_w = convert(Matrix{WT}, Q)
    P0_w = convert(Matrix{WT}, P0)
    ws.R_PD[] = PDMat(Symmetric(R_w))
    ws.Q_PD[] = PDMat(Symmetric(Q_w))
    ws.P0_PD[] = PDMat(Symmetric(P0_w))
    Rchol = ws.R_PD[].chol
    Qchol = ws.Q_PD[].chol
    P0chol = ws.P0_PD[].chol

    # tmp_RC = R^{-1} C
    copyto!(ws.tmp_RC, C)
    ldiv!(Rchol, ws.tmp_RC)
    copyto!(ws.C_inv_R, ws.tmp_RC')

    # tmp_QA = Q^{-1} A
    copyto!(ws.tmp_QA, A)
    ldiv!(Qchol, ws.tmp_QA)
    copyto!(ws.A_inv_Q, ws.tmp_QA')
    copyto!(ws.H_sub_entry, ws.tmp_QA)
    copyto!(ws.H_super_entry, ws.tmp_QA')

    # yt_given_xt = -C' * (R^{-1} C)
    mul!(ws.yt_given_xt, C', ws.tmp_RC)
    ws.yt_given_xt .*= -one(T)

    # xt_given_xt_1 = -Q^{-1}
    copyto!(ws.xt_given_xt_1, ws.I_mat)
    ldiv!(Qchol, ws.xt_given_xt_1)
    ws.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A' * (Q^{-1} A)
    mul!(ws.xt1_given_xt, A', ws.tmp_QA)
    ws.xt1_given_xt .*= -one(T)

    # x_t = -P0^{-1}
    copyto!(ws.x_t, ws.I_mat)
    ldiv!(P0chol, ws.x_t)
    ws.x_t .*= -one(T)

    return nothing
end

"""
    _copy_smooth_constants!(dst::SmoothWorkspace, src::SmoothWorkspace)

Copy all fields populated by `compute_smooth_constants!` from `src` to `dst`.
Used by the equal-length multi-trial fast path to amortize the constants —
`_precompute_shared_cov!` runs `compute_smooth_constants!` once on the
designated workspace, and each per-task worker copies into its own
`SmoothWorkspace` instead of recomputing the Cholesky factors and derived
terms per trial. The copies are pure `copyto!` over fixed-size matrices and
do not allocate.

Only the Gaussian-observation set of fields is copied; Poisson fits don't
go through the cov-cache fast path.
"""
function _copy_smooth_constants!(
    dst::SmoothWorkspace{T}, src::SmoothWorkspace{T}
) where {T<:Real}
    dst.R_PD[] = src.R_PD[]
    dst.Q_PD[] = src.Q_PD[]
    dst.P0_PD[] = src.P0_PD[]
    copyto!(dst.tmp_RC, src.tmp_RC)
    copyto!(dst.tmp_QA, src.tmp_QA)
    copyto!(dst.C_inv_R, src.C_inv_R)
    copyto!(dst.A_inv_Q, src.A_inv_Q)
    copyto!(dst.H_sub_entry, src.H_sub_entry)
    copyto!(dst.H_super_entry, src.H_super_entry)
    copyto!(dst.yt_given_xt, src.yt_given_xt)
    copyto!(dst.xt_given_xt_1, src.xt_given_xt_1)
    copyto!(dst.xt1_given_xt, src.xt1_given_xt)
    copyto!(dst.x_t, src.x_t)
    return dst
end

function compute_smooth_constants!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0

    # Wrap state-side covariances as PDMats (Poisson path doesn't need R
    # in this workspace path). See the Gaussian overload for the `convert`
    # rationale: it's a no-op when `WT === T` and avoids a Julia 1.10
    # `Cholesky` convert-method gap when `WT !== T` (ForwardDiff path).
    Q_w = convert(Matrix{WT}, Q)
    P0_w = convert(Matrix{WT}, P0)
    ws.Q_PD[] = PDMat(Symmetric(Q_w))
    ws.P0_PD[] = PDMat(Symmetric(P0_w))
    Q_chol = ws.Q_PD[].chol
    P0_chol = ws.P0_PD[].chol

    # Gradient terms: A_inv_Q = (Q_chol \ A)'
    tmp_QA = Q_chol \ A   # latent_dim × latent_dim
    copyto!(ws.A_inv_Q, tmp_QA')

    # Hessian block templates for state model
    copyto!(ws.H_sub_entry, tmp_QA)          # Q_chol \ A
    copyto!(ws.H_super_entry, tmp_QA')       # (Q_chol \ A)'

    # xt_given_xt_1 = -(Q_chol \ I) = -Q^{-1}
    copyto!(ws.xt_given_xt_1, ws.I_mat)
    ldiv!(Q_chol, ws.xt_given_xt_1)
    ws.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A' * (Q_chol \ A)
    mul!(ws.xt1_given_xt, A', tmp_QA)
    ws.xt1_given_xt .*= -one(T)

    # x_t = -(P0_chol \ I) = -P0^{-1}
    copyto!(ws.x_t, ws.I_mat)
    ldiv!(P0_chol, ws.x_t)
    ws.x_t .*= -one(T)

    return nothing
end

"""
    LDSConstantCache{T}

Cache of Cholesky-derived constants for a single LDS component used in SLDS smoothing.
This mirrors the *constant* parts of `SmoothWorkspace`, but does not include optimizer buffers
or block-tridiagonal storage.
"""
mutable struct LDSConstantCache{T<:Real}
    # PDMats for R, Q, P0. Rewrapped once per SLDS smoothing pass by
    # `compute_slds_constants!`; downstream consumers use `.chol.U` and
    # `logdet(...)`. Mirrors `SmoothWorkspace`'s `R_PD` / `Q_PD` / `P0_PD`.
    R_PD::Base.RefValue{PDMat{T,Matrix{T}}}
    Q_PD::Base.RefValue{PDMat{T,Matrix{T}}}
    P0_PD::Base.RefValue{PDMat{T,Matrix{T}}}

    # LL constant terms
    cP0::T
    cQ::T
    cR::T

    # Solve outputs / derived terms
    tmp_RC::Matrix{T}     # R^{-1}C  (obs_dim × latent_dim)
    tmp_QA::Matrix{T}     # Q^{-1}A  (latent_dim × latent_dim)

    C_inv_R::Matrix{T}    # (R^{-1}C)'  (latent_dim × obs_dim)
    A_inv_Q::Matrix{T}    # (Q^{-1}A)'  (latent_dim × latent_dim)

    # Hessian templates (state + Gaussian obs)
    H_sub_entry::Matrix{T}       # Q^{-1}A  (latent_dim × latent_dim)
    H_super_entry::Matrix{T}     # (Q^{-1}A)' (latent_dim × latent_dim)
    yt_given_xt::Matrix{T}       # -C'R^{-1}C  (latent_dim × latent_dim)  (Gaussian only; zero for Poisson)
    xt_given_xt_1::Matrix{T}     # -Q^{-1}
    xt1_given_xt::Matrix{T}      # -A'Q^{-1}A
    x_t::Matrix{T}               # -P0^{-1}

    # Model matrices needed for Poisson fast path
    C::Matrix{T}                 # obs_dim × latent_dim
    d::Vector{T}                 # obs_dim — Poisson log-link intercept (free in ℝ)
end

function LDSConstantCache(::Type{T}, latent_dim::Int, obs_dim::Int) where {T<:Real}
    return LDSConstantCache{T}(
        _pd_ref(PDMat(Matrix{T}(I, obs_dim, obs_dim))),         # R_PD placeholder
        _pd_ref(PDMat(Matrix{T}(I, latent_dim, latent_dim))),   # Q_PD placeholder
        _pd_ref(PDMat(Matrix{T}(I, latent_dim, latent_dim))),   # P0_PD placeholder
        zero(T),                                # cP0
        zero(T),                                # cQ
        zero(T),                                # cR
        zeros(T, obs_dim, latent_dim),          # tmp_RC
        zeros(T, latent_dim, latent_dim),       # tmp_QA
        zeros(T, latent_dim, obs_dim),          # C_inv_R  (latent_dim × obs_dim)
        zeros(T, latent_dim, latent_dim),       # A_inv_Q  (latent_dim × latent_dim)
        zeros(T, latent_dim, latent_dim),       # H_sub_entry
        zeros(T, latent_dim, latent_dim),       # H_super_entry
        zeros(T, latent_dim, latent_dim),       # yt_given_xt
        zeros(T, latent_dim, latent_dim),       # xt_given_xt_1
        zeros(T, latent_dim, latent_dim),       # xt1_given_xt
        zeros(T, latent_dim, latent_dim),       # x_t
        zeros(T, obs_dim, latent_dim),          # C
        zeros(T, obs_dim),                      # d
    )
end

"""
    compute_slds_constants!(cc, lds)

Fill a single-component cache with Cholesky-derived constants.
For Poisson observation models, `yt_given_xt` is left as zero and Poisson terms are handled per-iteration.
"""
function compute_slds_constants!(
    cc::LDSConstantCache{T}, lds::LinearDynamicalSystem{T,S,O}, I_mat::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0

    C = lds.obs_model.C
    cc.C .= C

    obs_dim, latent_dim = size(C)

    # Cache the Poisson log-link intercept directly (no exp — `d` is free in ℝ).
    # Gaussian observation models leave it zero (unused).
    if lds.obs_model isa PoissonObservationModel
        cc.d .= lds.obs_model.d
    else
        fill!(cc.d, zero(T))
    end

    # Wrap state-side covariances as PDMats. Observation R is wrapped further
    # below only on the Gaussian branch — Poisson leaves cc.R_PD on its
    # identity placeholder and `cc.cR` zero.
    cc.Q_PD[] = PDMat(Symmetrize!(Q))
    cc.P0_PD[] = PDMat(Symmetrize!(P0))
    Qchol = cc.Q_PD[].chol
    P0chol = cc.P0_PD[].chol

    # tmp_QA = Q^{-1}A, A_inv_Q = (Q^{-1}A)'
    copyto!(cc.tmp_QA, A)
    ldiv!(Qchol, cc.tmp_QA)
    copyto!(cc.A_inv_Q, cc.tmp_QA')
    copyto!(cc.H_sub_entry, cc.tmp_QA)
    copyto!(cc.H_super_entry, cc.tmp_QA')

    # xt_given_xt_1 = -Q^{-1}
    copyto!(cc.xt_given_xt_1, I_mat)
    ldiv!(Qchol, cc.xt_given_xt_1)
    cc.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A'Q^{-1}A = -A' * (Q^{-1}A)
    mul!(cc.xt1_given_xt, A', cc.tmp_QA)
    cc.xt1_given_xt .*= -one(T)

    # x_t = -P0^{-1}
    copyto!(cc.x_t, I_mat)
    ldiv!(P0chol, cc.x_t)
    cc.x_t .*= -one(T)

    # Observation terms only if Gaussian
    if lds.obs_model isa GaussianObservationModel{T}
        cc.R_PD[] = PDMat(Symmetrize!(lds.obs_model.R))
        Rchol = cc.R_PD[].chol

        # tmp_RC = R^{-1}C, C_inv_R = (R^{-1}C)'
        copyto!(cc.tmp_RC, C)
        ldiv!(Rchol, cc.tmp_RC)
        copyto!(cc.C_inv_R, cc.tmp_RC')

        # yt_given_xt = -C'R^{-1}C
        mul!(cc.yt_given_xt, C', cc.tmp_RC)
        cc.yt_given_xt .*= -one(T)
    else
        fill!(cc.yt_given_xt, zero(T))
        fill!(cc.C_inv_R, zero(T))  # unused for Poisson
    end

    # LL normalizers from cached Cholesky factors.
    cc.cP0 = -T(0.5) * (T(latent_dim) * log(T(2π)) + logdet(cc.P0_PD[]))
    cc.cQ = -T(0.5) * (T(latent_dim) * log(T(2π)) + logdet(cc.Q_PD[]))

    if lds.obs_model isa GaussianObservationModel{T}
        cc.cR = -T(0.5) * (T(obs_dim) * log(T(2π)) + logdet(cc.R_PD[]))
    else
        cc.cR = zero(T)
    end

    return nothing
end

"""
    SLDSSmoothWorkspace{T}

Workspace for SLDS smoothing that matches the LDS backend shape:
- Owns a BlockTridiagonalWorkspace (H blocks + sparse + inverse scratch)
- Owns per-component LDSConstantCache objects
- Owns optimizer buffers (X₀, grad, sparse Hessian template)
- Owns Poisson temporary vectors
"""
struct SLDSSmoothWorkspace{T<:Real}
    btd::BlockTridiagonalWorkspace{T}
    I_mat::Matrix{T}

    consts::Vector{LDSConstantCache{T}}

    # Optim buffers
    X₀::Vector{T}
    grad_buf::Matrix{T}
    grad_vec::Vector{T}
    initial_h::SparseMatrixCSC{T,Int}

    # Poisson temporaries (shared)
    z::Vector{T}   # obs_dim
    λ::Vector{T}   # obs_dim

    # temp vectors
    dxt::Vector{T}        # latent_dim
    dxt_next::Vector{T}   # latent_dim
    dyt::Vector{T}        # obs_dim  (Gaussian only)
    tmp1::Vector{T}       # latent_dim
    tmp2::Vector{T}       # latent_dim
    tmp3::Vector{T}       # latent_dim

    # LL buffers
    ll_vec::Vector{T}   # accumulator (length tsteps)
    ll_tmp::Vector{T}   # per-component scratch (length tsteps)
end

function SLDSSmoothWorkspace(::Type{T}, slds::SLDS, tsteps::Int) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    K = length(slds.LDSs)

    btd = BlockTridiagonalWorkspace(T, latent_dim, tsteps)
    I_mat = Matrix{T}(I, latent_dim, latent_dim)

    consts = [LDSConstantCache(T, latent_dim, obs_dim) for _ in 1:K]

    X₀ = zeros(T, latent_dim * tsteps)
    grad_buf = zeros(T, latent_dim, tsteps)
    grad_vec = zeros(T, latent_dim * tsteps)
    initial_h = copy(btd.H_sparse)

    z = zeros(T, obs_dim)
    λ = zeros(T, obs_dim)

    dxt = zeros(T, latent_dim)
    dxt_next = zeros(T, latent_dim)
    dyt = zeros(T, obs_dim)
    tmp1 = zeros(T, latent_dim)
    tmp2 = zeros(T, latent_dim)
    tmp3 = zeros(T, latent_dim)

    ll_vec = zeros(T, tsteps)
    ll_tmp = zeros(T, tsteps)

    ws = SLDSSmoothWorkspace{T}(
        btd,
        I_mat,
        consts,
        X₀,
        grad_buf,
        grad_vec,
        initial_h,
        z,
        λ,
        dxt,
        dxt_next,
        dyt,
        tmp1,
        tmp2,
        tmp3,
        ll_vec,
        ll_tmp,
    )

    # Cache constants once
    for k in 1:K
        compute_slds_constants!(ws.consts[k], slds.LDSs[k], ws.I_mat)
    end

    return ws
end

"""
Refresh the per-regime constant caches after an M-step has updated the LDS parameters.
Must be called before the next E-step so that Cholesky factors, Hessian templates, etc.
reflect the current Q, R, A, P0.
"""
function refresh_slds_constants!(ws::SLDSSmoothWorkspace{T}, slds) where {T}
    for k in eachindex(slds.LDSs)
        compute_slds_constants!(ws.consts[k], slds.LDSs[k], ws.I_mat)
    end
    return nothing
end

"""
    KalmanWorkspace{T<:Real}

Pre-allocated workspace for the information-form Kalman filter + RTS smoother path
(ported from StateSpaceAnalysis). Retained for the marginal log-likelihood and
future particle-filter use; no longer used as a `fit!` E-step backend.

The workspace is split into three layers:

1. **Shared covariance storage** (`pred_cov`, `filt_cov`, `pred_icov`, `smooth_cov`,
   `G`). Populated once per E-step by the covariance forward-backward pass — these
   quantities depend only on `A, Q, C, R, P0`, not on the observations, so they are
   reused across all trials. Stored as `Vector{PDMat}` to keep Cholesky factors cached.

2. **Shared 3D aliases** (`p_smooth_shared`, `p_smooth_tt1_shared`). Dense arrays that
   each trial's `FilterSmooth.p_smooth` / `FilterSmooth.p_smooth_tt1` is set to point
   at (same Julia object, not a copy). `sufficient_statistics!` only reads these
   fields, so reference-sharing is safe.

3. **Per-trial mean-pass buffers** (`pred_mean`, `filt_mean`, `smooth_mean`, `Bu`,
   `CiRY`, `y_minus_d`). Shape `(D, T, ntrials)` — threads write into disjoint trial
   slices, so no locking is required.

`ux_dim` is 0 when the `B` input is absent.
"""
struct KalmanWorkspace{T<:Real}

    # constants
    latent_dim::Int
    obs_dim::Int
    state_input_dim::Int
    obs_input_dim::Int
    tsteps::Int
    ntrials::Int

    # Shared covariance storage (populated once per E-step)
    pred_cov::Vector{PDMat{T,Matrix{T}}}
    filt_cov::Vector{PDMat{T,Matrix{T}}}
    pred_icov::Vector{PDMat{T,Matrix{T}}}
    smooth_cov::Vector{PDMat{T,Matrix{T}}}
    sum_smooth_cov_all::Matrix{T}   # (D, D) - covariance P_smooth[t]
    sum_smooth_cov_prev::Matrix{T}   # (D, D) - covariance P_smooth[t]
    sum_smooth_cov_next::Matrix{T}   # (D, D) - covariance P_smooth[t]
    sum_smooth_xcov::Matrix{T}  # (D, D) - cross-covariance P_smooth[t, t-1]
    G::Array{T,3}  # (D, D, T-1)

    # Pre-allocated scratch buffers for covariance_forward_backward! (no per-step allocs)
    cov_tmp1::Matrix{T}   # (D, D)
    cov_tmp2::Matrix{T}   # (D, D)
    pd_tmp::Base.RefValue{PDMat{T,Matrix{T}}} #(D, D)
    obs_pd_tmp::Base.RefValue{PDMat{T,Matrix{T}}} #(p, p)

    # Per-trial D-vector scratch for _filter_mean_trial! (column n used by trial n)
    mean_tmp::Matrix{T}   # (D, ntrials)

    # Shared 3D aliases for FilterSmooth reference-sharing
    p_smooth_shared::Array{T,3}      # (D, D, T)
    p_smooth_tt1_shared::Array{T,3}  # (D, D, T)

    # Derived constants (refreshed each E-step; M-step mutates Q, R, P0)
    Q_PD::Base.RefValue{PDMat{T,Matrix{T}}}
    P0_PD::Base.RefValue{PDMat{T,Matrix{T}}}
    R_PD::Base.RefValue{PDMat{T,Matrix{T}}}

    #  Priors — matrix-normal halves stored as `MNPrior`s (M₀ + Λ);
    #  inverse-Wishart halves are split into (μ, df) pairs to keep the
    #  hot-path arithmetic indirection-free.
    P0_mu::Matrix{T}
    P0_df::Int

    AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
    Q_mu::Matrix{T}
    Q_df::Int

    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
    R_mu::Matrix{T}
    R_df::Int

    # Derived terms for Kalman gain and LL computation (refreshed each E-step)
    CiR::Matrix{T}                              # C' * R^{-1}   (D × p)
    CiRC::Base.RefValue{PDMat{T,Matrix{T}}}     # C' * R^{-1} * C  (D × D), symmetric
    shared_entropy::Base.RefValue{T}

    # Per-trial mean-pass buffers (thread-safe via trial-slice access)
    pred_mean::Array{T,3}    # (D, T, ntrials)
    filt_mean::Array{T,3}    # (D, T, ntrials)
    smooth_mean::Array{T,3}  # (D, T, ntrials)
    Bu::Array{T,3}           # (D, T, ntrials)
    CiRY::Array{T,3}         # (D, T, ntrials)
    y_minus_d::Array{T,3}    # (p, T, ntrials)
    innovation::Array{T,3}   # (p, T, ntrials)

    x_prev::Matrix{T}
    x_next::Matrix{T}
    x_init::Matrix{T}
    x_cur::Matrix{T}

    # Sufficient-statistics scratch (reused each E-step; bottom-right uu/dd
    # block is initialized once from `initialize_SufficientStatistics` and not
    # mutated thereafter).
    dyn_xx_buf::Matrix{T}    # ((D + ux_dim) × (D + ux_dim))
    obs_xx_buf::Matrix{T}    # ((D + uy_dim) × (D + uy_dim))
end

"""
    KalmanWorkspace(lds::LinearDynamicalSystem, tsteps::Int, ntrials::Int)

Allocate a `KalmanWorkspace` sized for the given `lds` and data shape. Requires
`lds.obs_model isa GaussianObservationModel`.
"""
@views function KalmanWorkspace(
    lds::LinearDynamicalSystem{T,S,O}, tsteps::Int, ntrials::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    state_input_dim = size(lds.state_model.B, 2)
    obs_input_dim = size(lds.obs_model.D, 2)

    # Placeholder PDMats; re-wrapped at the start of every E-step
    placeholder_D = PDMat(Matrix{T}(I, D, D))
    placeholder_p = PDMat(Matrix{T}(I, p, p))

    # MN priors stored verbatim; no PDMat wrapping (the M-step needs (XX + Λ)
    # solves where XX changes every iteration, so a cached chol of Λ-alone
    # would not be reused).
    placeholder_AB = lds.state_model.AB_prior
    placeholder_CD = lds.obs_model.CD_prior

    if isnothing(lds.state_model.P0_prior)
        placeholder_P0_mu = zeros(T, D, D)
        placeholder_P0_df = 0
    else
        placeholder_P0_mu = Matrix{T}(lds.state_model.P0_prior.Ψ)
        placeholder_P0_df = lds.state_model.P0_prior.ν
    end

    if isnothing(lds.state_model.Q_prior)
        placeholder_Q_mu = zeros(T, D, D)
        placeholder_Q_df = 0
    else
        placeholder_Q_mu = Matrix{T}(lds.state_model.Q_prior.Ψ)
        placeholder_Q_df = lds.state_model.Q_prior.ν
    end

    if isnothing(lds.obs_model.R_prior)
        placeholder_R_mu = zeros(T, p, p)
        placeholder_R_df = 0
    else
        placeholder_R_mu = Matrix{T}(lds.obs_model.R_prior.Ψ)
        placeholder_R_df = lds.obs_model.R_prior.ν
    end

    pred_cov = [PDMat(Matrix{T}(I, D, D)) for _ in 1:tsteps]
    pred_icov = [PDMat(Matrix{T}(I, D, D)) for _ in 1:tsteps]
    filt_cov = [PDMat(Matrix{T}(I, D, D)) for _ in 1:tsteps]
    smooth_cov = [PDMat(Matrix{T}(I, D, D)) for _ in 1:tsteps]
    G = zeros(T, D, D, max(tsteps - 1, 0))

    p_smooth_shared = zeros(T, D, D, tsteps)
    p_smooth_tt1_shared = zeros(T, D, D, tsteps)

    return KalmanWorkspace{T}(
        D,                              # latent_dim
        p,                              # obs_dim
        state_input_dim,                # control_dim
        obs_input_dim,                  # observation_control_dim
        tsteps,                         # time_steps
        ntrials,                        # number_of_trials
        pred_cov,                       # Vector{PDMat} for P_pred[t]
        filt_cov,                       # Vector{PDMat} for P_filt[t]
        pred_icov,                      # Vector{PDMat} for P_pred[t]^{-1} as PDMats (cached Cholesky factors)
        smooth_cov,                     # Vector{PDMat} for P_smooth[t]
        zeros(T, D, D),                 # sum_smooth_cov_all, Matrix{T}  for sum P_smooth[1:T]
        zeros(T, D, D),                 # sum_smooth_cov_prev, Matrix{T}  for sum P_smooth[1:(T-1)]
        zeros(T, D, D),                 # sum_smooth_cov_next, Matrix{T}  for sum P_smooth[2:T]
        zeros(T, D, D),                 # sum_smooth_xcov, Matrix{T} for sum P_smooth[t, t-1]
        G,                              # Array{T,3} for G[t] = P_filt[t] * A' * P_pred[t+1]^{-1}
        zeros(T, D, D),                 # cov_tmp1
        zeros(T, D, D),                 # cov_tmp2
        _pd_ref(placeholder_D),             # pd_tmp
        _pd_ref(placeholder_p),             # obs_pd_tmp
        zeros(T, D, ntrials),           # mean_tmp
        p_smooth_shared,                # (D, D, T)
        p_smooth_tt1_shared,            # (D, D, T)
        _pd_ref(placeholder_D),             # Q_PD
        _pd_ref(placeholder_D),             # P0_PD
        _pd_ref(placeholder_p),             # R_PD
        placeholder_P0_mu,              # P0_mu
        placeholder_P0_df,              # P0_df
        placeholder_AB,                 # AB_prior
        placeholder_Q_mu,               # Q_mu
        placeholder_Q_df,               # Q_df
        placeholder_CD,                 # CD_prior
        placeholder_R_mu,               # R_mu
        placeholder_R_df,               # R_df
        zeros(T, D, p),                 # CiR
        _pd_ref(placeholder_D),             # CiRC
        Ref(zero(T)),                   # shared_entropy
        zeros(T, D, tsteps, ntrials),   # pred_mean
        zeros(T, D, tsteps, ntrials),   # filt_mean
        zeros(T, D, tsteps, ntrials),   # smooth_mean
        zeros(T, D, tsteps, ntrials),   # Bu
        zeros(T, D, tsteps, ntrials),   # CiRY
        zeros(T, p, tsteps, ntrials),   # y_minus_d
        zeros(T, p, tsteps, ntrials),   # innovations
        zeros(T, D, (tsteps - 1) * ntrials),# x_prev
        zeros(T, D, (tsteps - 1) * ntrials),# x_next
        zeros(T, D, ntrials),           # x_init
        zeros(T, D, tsteps * ntrials),    # x_cur
        zeros(T, D + 1 + state_input_dim, D + 1 + state_input_dim),  # dyn_xx_buf
        zeros(T, D + 1 + obs_input_dim, D + 1 + obs_input_dim),      # obs_xx_buf
    )
end

# =============================================================================
# Model ⇄ workspace glue: FilterSmooth construction and parameter (un)packing.
# (Absorbed from the former common.jl.)
# =============================================================================

function _extract_state_params(state_model::GaussianStateModel{T}) where {T}
    return (
        A=state_model.A,
        B=state_model.B,
        Q=state_model.Q,
        b=state_model.b,
        x0=state_model.x0,
        P0=state_model.P0,
    )
end

"""
    initialize_FilterSmooth(model, tsteps::Int)

Initialize a per-trial `FilterSmooth` buffer sized for `tsteps` timesteps.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, tsteps::Int; cov_alias::Bool=false
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = model.latent_dim
    if cov_alias
        p_smooth = zeros(T, 0, 0, 0)
        p_smooth_tt1 = zeros(T, 0, 0, 0)
        E_zz = zeros(T, 0, 0, 0)
        E_zz_prev = zeros(T, 0, 0, 0)
    else
        p_smooth = zeros(T, D, D, tsteps)
        p_smooth_tt1 = zeros(T, D, D, tsteps)
        E_zz = zeros(T, D, D, tsteps)
        E_zz_prev = zeros(T, D, D, tsteps)
    end
    return FilterSmooth{T}(
        zeros(T, D, tsteps),       # x_smooth
        p_smooth,
        p_smooth_tt1,
        zeros(T, D, tsteps),       # E_z
        E_zz,
        E_zz_prev,
        zero(T),                   # entropy
    )
end

"""
    initialize_FilterSmooth(model, tsteps_per_trial::AbstractVector{<:Integer};
                            cov_alias=false)

Initialize a `TrialFilterSmooth` with one `FilterSmooth` per trial. Trial lengths
may differ (but don't have to).

Set `cov_alias=true` only when the caller knows the cov-cache fast path will
run (equal-length multi-trial Gaussian via `_fit_tridiag!`) — in that case
every per-trial `p_smooth` / `p_smooth_tt1` is allocated as a `(0, 0, 0)` stub
because `smooth!` aliases them to `sws.p_smooth_shared` on every E-step. The
SLDS / Poisson / ragged paths invoke the per-trial smoother directly and
write into `fs.p_smooth`, so they must keep the default `cov_alias=false`.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    cov_alias::Bool=false,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    # if tsteps_per_trial has varying lengths, we can't alias the cov caches to a shared zero-array
    if cov_alias && length(unique(tsteps_per_trial)) != 1
        throw(
            ArgumentError(
                "cov_alias=true is only valid when all trials have the same number of timesteps; got tsteps_per_trial=$(tsteps_per_trial)",
            ),
        )
    end
    filter_smooths = [
        initialize_FilterSmooth(model, Int(t); cov_alias=cov_alias) for
        t in tsteps_per_trial
    ]
    return TrialFilterSmooth(filter_smooths)
end

function _extract_obs_params(obs_model::GaussianObservationModel{T}) where {T}
    return (C=obs_model.C, R=obs_model.R, d=obs_model.d, D=obs_model.D)
end

function _extract_obs_params(obs_model::PoissonObservationModel{T}) where {T}
    return (C=obs_model.C, d=obs_model.d)
end
