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
PDMats 0.11.40 added a third (Cholesky) type parameter to `PDMat`, which turns
the two-parameter `PDMat{T,Matrix{T}}` into a non-concrete UnionAll. 
TODO: Consider lowerbounding PDMats to 0.11.40 and inlining the 3-param alias.
=#
if length(Base.unwrap_unionall(PDMat).parameters) == 3
    const DensePDMat{T} = PDMat{T,Matrix{T},Cholesky{T,Matrix{T}}}
else
    const DensePDMat{T} = PDMat{T,Matrix{T}}
end

_pd_ref(pd::PDMat) = Base.RefValue{DensePDMat{eltype(pd)}}(pd)

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

    #=
    initial conditions. `init_n` is the effective sample count (e.g.
    `ntrials` for unweighted fits; `Σₙ w[n,1]` for SLDS-style soft
    responsibility weights). Stored as `T` rather than `Int` so the
    weighted aggregator can flow non-integer counts through the M-step
    without truncation.
    =#
    init_n::T
    init_xy::Matrix{T}
    #= Raw scatter `Σγ(x₁x₁' + P₁)`, only ever read as a plain matrix. Not a PDMat:
    a zero-weight regime makes it the zero matrix (would fail the Cholesky). =#
    init_yy::Base.RefValue{Matrix{T}}

    # transitions model
    dyn_n::T
    dyn_xx::Base.RefValue{DensePDMat{T}}
    dyn_xy::Matrix{T}
    dyn_yy::Base.RefValue{DensePDMat{T}}

    # observation model
    obs_n::T
    obs_xx::Base.RefValue{DensePDMat{T}}
    obs_xy::Matrix{T}
    obs_yy::Base.RefValue{DensePDMat{T}}
end

"""
    SmoothConstants{T<:Real}

Cholesky-derived constants for one LDS: PDMat-wrapped covariances, cached
log-likelihood normalizers, and the derived gradient / Hessian block
templates. Everything here depends only on the model parameters
(`A, Q, C, R, P0`), not on the data or the latent iterate.

Filled by `compute_smooth_constants!` — once per E-step on the single-LDS
path (where one lives inside each `SmoothWorkspace`), or once per regime per
smoothing pass on the SLDS path (where `SLDSSmoothWorkspace` owns a vector of
them, one per component).

Mutable so the PDMat wrappers and scalar normalizers can be reassigned; the
buffer fields are `const` (their *contents* are overwritten in place).
"""
mutable struct SmoothConstants{T<:Real}
    # PDMat-wrapped covariances (each caches its own Cholesky; consumed via
    # `cc.X_PD.chol.U` for triangular solves and `logdet(cc.X_PD)`).
    R_PD::DensePDMat{T}      # (obs_dim × obs_dim)
    Q_PD::DensePDMat{T}      # (latent_dim × latent_dim)
    P0_PD::DensePDMat{T}     # (latent_dim × latent_dim)

    # Cached log-likelihood normalizers -0.5*(dim*log(2π) + logdet(Σ))
    cP0::T
    cQ::T
    cR::T

    # Solve outputs
    const tmp_RC::Matrix{T}    # obs_dim × latent_dim   (R^{-1} C)
    const tmp_QA::Matrix{T}    # latent_dim × latent_dim (Q^{-1} A)

    # Derived terms for Gradient
    const C_inv_R::Matrix{T}   # (R_chol \ C)' = C'inv(R), size (latent_dim × obs_dim)
    const A_inv_Q::Matrix{T}   # (Q_chol \ A)' = A'inv(Q), size (latent_dim × latent_dim)

    # Derived terms for Hessian block templates
    const H_sub_entry::Matrix{T}    # Q_chol \ A, size (latent_dim × latent_dim)
    const H_super_entry::Matrix{T}  # H_sub_entry', size (latent_dim × latent_dim)
    const yt_given_xt::Matrix{T}    # -C'*(R_chol \ C), size (latent_dim × latent_dim)
    const xt_given_xt_1::Matrix{T}  # -(Q_chol \ I), size (latent_dim × latent_dim)
    const xt1_given_xt::Matrix{T}   # -A'*(Q_chol \ A), size (latent_dim × latent_dim)
    const x_t::Matrix{T}            # -(P0_chol \ I), size (latent_dim × latent_dim)

    # Identity scratch for the `chol \ I` solves above
    const I_mat::Matrix{T}          # (latent_dim × latent_dim)
end

function SmoothConstants(::Type{T}, latent_dim::Int, obs_dim::Int) where {T<:Real}
    # Placeholder PDMats — rewrapped by `compute_smooth_constants!`.
    return SmoothConstants{T}(
        PDMat(Matrix{T}(I, obs_dim, obs_dim)),          # R_PD
        PDMat(Matrix{T}(I, latent_dim, latent_dim)),    # Q_PD
        PDMat(Matrix{T}(I, latent_dim, latent_dim)),    # P0_PD
        zero(T),                                        # cP0
        zero(T),                                        # cQ
        zero(T),                                        # cR
        zeros(T, obs_dim, latent_dim),                  # tmp_RC
        zeros(T, latent_dim, latent_dim),               # tmp_QA
        zeros(T, latent_dim, obs_dim),                  # C_inv_R
        zeros(T, latent_dim, latent_dim),               # A_inv_Q
        zeros(T, latent_dim, latent_dim),               # H_sub_entry
        zeros(T, latent_dim, latent_dim),               # H_super_entry
        zeros(T, latent_dim, latent_dim),               # yt_given_xt
        zeros(T, latent_dim, latent_dim),               # xt_given_xt_1
        zeros(T, latent_dim, latent_dim),               # xt1_given_xt
        zeros(T, latent_dim, latent_dim),               # x_t
        Matrix{T}(I, latent_dim, latent_dim),           # I_mat
    )
end

"""
    NewtonBuffers{T<:Real}

Per-trial Newton-smoother scratch: the vectorized iterate / gradient, plus the
small temp vectors used by the `gradient!` and `joint_loglikelihood!` kernels.
Owned as `opt` by both `SmoothWorkspace` and `SLDSSmoothWorkspace`, so kernels
shared between the two paths address scratch through one set of field paths.
"""
struct NewtonBuffers{T<:Real}
    X0::Vector{T}             # Vectorized latent path (latent_dim * tsteps)
    grad_buf::Matrix{T}       # Gradient output buffer (latent_dim × tsteps)
    grad_vec::Vector{T}       # Vectorized gradient (latent_dim * tsteps)

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
end

function NewtonBuffers(
    ::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int
) where {T<:Real}
    return NewtonBuffers{T}(
        zeros(T, latent_dim * tsteps),  # X0
        zeros(T, latent_dim, tsteps),   # grad_buf
        zeros(T, latent_dim * tsteps),  # grad_vec
        zeros(T, latent_dim),           # dxt
        zeros(T, latent_dim),           # dxt_next
        zeros(T, obs_dim),              # dyt
        zeros(T, latent_dim),           # tmp1
        zeros(T, latent_dim),           # tmp2
        zeros(T, latent_dim),           # tmp3
        zeros(T, tsteps),               # ll_vec
        zeros(T, latent_dim),           # temp_dx
        zeros(T, obs_dim),              # temp_dy
        zeros(T, latent_dim),           # temp_solve_Q
        zeros(T, obs_dim),              # temp_solve_R
    )
end

"""
    RegressionBuffers{T<:Real}

Regression-shaped scratch sized for `[A b B]` (dyn_reg_dim = D + 1 + ux_dim)
and `[C d D]` (obs_reg_dim = D + 1 + uy_dim). Used by the suf-based M-step
updates, reused as accumulators by the TD sufficient-stats aggregators, and
(`CD`/`Syz`) as the per-chunk gradient accumulators of the Poisson emission
M-step.
"""
struct RegressionBuffers{T<:Real}
    Sxz::Matrix{T}            # (latent_dim × dyn_reg_dim) for update_A_b!
    Szz_Ab::Matrix{T}         # (dyn_reg_dim × dyn_reg_dim) for update_A_b!
    AB::Matrix{T}             # (latent_dim × dyn_reg_dim) for update_A_b!
    Syz::Matrix{T}            # (obs_dim × obs_reg_dim) for update_C_d!
    Szz_Cd::Matrix{T}         # (obs_reg_dim × obs_reg_dim) for update_C_d!
    CD::Matrix{T}             # (obs_dim × obs_reg_dim) for update_C_d!
    Q_sum::Matrix{T}          # (latent_dim × latent_dim)
    R_sum::Matrix{T}          # (obs_dim × obs_dim)
    S0_sum::Matrix{T}         # (latent_dim × latent_dim)
end

function RegressionBuffers(
    ::Type{T}, latent_dim::Int, obs_dim::Int; ux_dim::Int=0, uy_dim::Int=0
) where {T<:Real}
    #=
    The "+1" is for the affine bias column (b for the dynamics regression,
    d for the observation regression); ux_dim / uy_dim add the user input
    columns when controls are supplied.
    =#
    dyn_reg_dim = latent_dim + 1 + ux_dim
    obs_reg_dim = latent_dim + 1 + uy_dim
    return RegressionBuffers{T}(
        zeros(T, latent_dim, dyn_reg_dim),   # Sxz
        zeros(T, dyn_reg_dim, dyn_reg_dim),  # Szz_Ab
        zeros(T, latent_dim, dyn_reg_dim),   # AB
        zeros(T, obs_dim, obs_reg_dim),      # Syz
        zeros(T, obs_reg_dim, obs_reg_dim),  # Szz_Cd
        zeros(T, obs_dim, obs_reg_dim),      # CD
        zeros(T, latent_dim, latent_dim),    # Q_sum
        zeros(T, obs_dim, obs_dim),          # R_sum
        zeros(T, latent_dim, latent_dim),    # S0_sum
    )
end

"""
    ElboBuffers{T<:Real}

Accumulator / work buffers for the `Q_state!` / `Q_obs!` ELBO terms (Gaussian
and Poisson variants). A few double as free scratch in the M-step covariance
updates (`temp`, `obs_temp`, `obs_work`).
"""
struct ElboBuffers{T<:Real}
    # Q_state buffers
    temp::Matrix{T}           # (latent_dim × latent_dim) - main accumulator
    sum_E_zz::Matrix{T}       # (latent_dim × latent_dim)
    sum_E_zzm1::Matrix{T}     # (latent_dim × latent_dim)
    sum_E_cross::Matrix{T}    # (latent_dim × latent_dim)
    sum_mu_t::Vector{T}       # (latent_dim,)
    sum_mu_tm1::Vector{T}     # (latent_dim,)
    temp2::Matrix{T}          # (latent_dim × latent_dim) - for A * sum_E_zzm1 * A'

    # Q_obs buffers (Gaussian)
    obs_temp::Matrix{T}       # (obs_dim × obs_dim) - accumulator
    obs_work::Matrix{T}       # (obs_dim × obs_dim) - work matrix
    ytil::Vector{T}           # (obs_dim,) - residualized y
    sum_yy::Matrix{T}         # (obs_dim × obs_dim)
    sum_yz::Matrix{T}         # (obs_dim × latent_dim)
    obs_work1::Matrix{T}      # (obs_dim × obs_dim)
    obs_work2::Matrix{T}      # (latent_dim × obs_dim)

    # Q_obs buffers (Poisson)
    h_obs::Vector{T}          # (obs_dim,) - h_t = C * E[x_t] + d
    rho_obs::Vector{T}        # (obs_dim,) - variance correction
    CP_obs::Matrix{T}         # (obs_dim × latent_dim) - C * P_t
    CEz_obs::Vector{T}        # (obs_dim,) - C * E[x_t]
end

function ElboBuffers(::Type{T}, latent_dim::Int, obs_dim::Int) where {T<:Real}
    return ElboBuffers{T}(
        zeros(T, latent_dim, latent_dim),  # temp
        zeros(T, latent_dim, latent_dim),  # sum_E_zz
        zeros(T, latent_dim, latent_dim),  # sum_E_zzm1
        zeros(T, latent_dim, latent_dim),  # sum_E_cross
        zeros(T, latent_dim),              # sum_mu_t
        zeros(T, latent_dim),              # sum_mu_tm1
        zeros(T, latent_dim, latent_dim),  # temp2
        zeros(T, obs_dim, obs_dim),        # obs_temp
        zeros(T, obs_dim, obs_dim),        # obs_work
        zeros(T, obs_dim),                 # ytil
        zeros(T, obs_dim, obs_dim),        # sum_yy
        zeros(T, obs_dim, latent_dim),     # sum_yz
        zeros(T, obs_dim, obs_dim),        # obs_work1
        zeros(T, latent_dim, obs_dim),     # obs_work2
        zeros(T, obs_dim),                 # h_obs
        zeros(T, obs_dim),                 # rho_obs
        zeros(T, obs_dim, latent_dim),     # CP_obs
        zeros(T, obs_dim),                 # CEz_obs
    )
end

"""
    TDAggBuffers{T<:Real}

Buffers for the TD sufficient-statistics aggregator plus the shared
smoothed-covariance storage of the equal-length multi-trial fast path.

`p_smooth_shared` / `p_smooth_tt1_shared`: the BT Hessian (and therefore its
inverse) is observation-independent; when all trials of a fit share the same
length, the smoothed covariances are computed once on a designated workspace
and aliased by every trial's `FilterSmooth.p_smooth` / `p_smooth_tt1` field.

The `*_const` blocks are data-only aggregates filled once at fit entry
(`_td_init_const_blocks!`); the rest are overwritten each E-step by
`_aggregate_td_suff_stats!`.
"""
struct TDAggBuffers{T<:Real}
    # Shared smoothed-covariance storage (equal-length fast path)
    p_smooth_shared::Array{T,3}      # (latent_dim, latent_dim, tsteps)
    p_smooth_tt1_shared::Array{T,3}  # (latent_dim, latent_dim, tsteps)

    # Aggregator output buffers, shaped to match `SufficientStatistics`
    init_xy::Matrix{T}               # (1, latent_dim)   Σₙ x_init
    dyn_xy::Matrix{T}                # (dyn_reg_dim, D)  Σₙ Σₜ [x_{t-1};1;u_{t-1}] xₜ'
    obs_xy::Matrix{T}                # (obs_reg_dim, p)  Σₙ Σₜ [xₜ;1;vₜ] yₜ'

    # Cross-trial smoothed-covariance accumulators
    sum_smooth_cov_prev::Matrix{T}   # (D, D)  Σₙ Σ_{t=1:Tₙ-1} P_smooth[t]
    sum_smooth_cov_next::Matrix{T}   # (D, D)  Σₙ Σ_{t=2:Tₙ}   P_smooth[t]
    sum_smooth_cov_all::Matrix{T}    # (D, D)  Σₙ Σ_{t=1:Tₙ}   P_smooth[t]
    sum_smooth_xcov::Matrix{T}       # (D, D)  Σₙ Σ_{t=2:Tₙ}   P_smooth_tt1[t]

    #=
    Constant aggregates over the input data (filled once at fit entry, not
    touched again). The y-only / uy-only blocks of obs_xx, obs_xy, obs_yy and
    the ux-only blocks of dyn_xx are observation-independent so we cache them
    here to skip re-summing every E-step.
    =#
    obs_yy_const::Matrix{T}          # (p, p)                Σₙ Σₜ yₜ yₜ'
    obs_xy_const::Matrix{T}          # (obs_reg_dim, p)      bias + uy-rows of obs_xy
    obs_xx_const::Matrix{T}          # (obs_reg_dim, obs_reg_dim) bias / uy blocks
    dyn_xx_const::Matrix{T}          # (dyn_reg_dim, dyn_reg_dim) bias / ux blocks
end

function TDAggBuffers(
    ::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int; ux_dim::Int=0, uy_dim::Int=0
) where {T<:Real}
    dyn_reg_dim = latent_dim + 1 + ux_dim
    obs_reg_dim = latent_dim + 1 + uy_dim
    return TDAggBuffers{T}(
        zeros(T, latent_dim, latent_dim, tsteps),  # p_smooth_shared
        zeros(T, latent_dim, latent_dim, tsteps),  # p_smooth_tt1_shared
        zeros(T, 1, latent_dim),                   # init_xy
        zeros(T, dyn_reg_dim, latent_dim),         # dyn_xy
        zeros(T, obs_reg_dim, obs_dim),            # obs_xy
        zeros(T, latent_dim, latent_dim),          # sum_smooth_cov_prev
        zeros(T, latent_dim, latent_dim),          # sum_smooth_cov_next
        zeros(T, latent_dim, latent_dim),          # sum_smooth_cov_all
        zeros(T, latent_dim, latent_dim),          # sum_smooth_xcov
        zeros(T, obs_dim, obs_dim),                # obs_yy_const
        zeros(T, obs_reg_dim, obs_dim),            # obs_xy_const
        zeros(T, obs_reg_dim, obs_reg_dim),        # obs_xx_const
        zeros(T, dyn_reg_dim, dyn_reg_dim),        # dyn_xx_const
    )
end

"""
    BatchedBuffers{T<:Real}

Batched mean-pass buffers for the equal-length cov-cache fast path. Only the
designated `sws_pool[1]` workspace carries one (constructed with the fit's
actual `ntrials`); the rest of the pool has `batched = nothing`. The
`(D, T, N)` tensors share storage with their `(D*T, N)` reshaped views used
as matrix RHS for `block_tridiagonal_backsubst!`.

The stacked data tensors (`y` / `ux` / `uy`) are populated once at the first
batched `smooth!` call (data is constant across EM iters within a fit);
`data_valid` flips to `true` after that populate.
"""
struct BatchedBuffers{T<:Real}
    x_mat::Array{T,3}         # (latent_dim, tsteps, ntrials) - current iterate
    grad_buf::Array{T,3}      # (latent_dim, tsteps, ntrials) - gradient! output
    dxt::Matrix{T}            # (latent_dim, ntrials)
    dxt_next::Matrix{T}       # (latent_dim, ntrials)
    dyt::Matrix{T}            # (obs_dim, ntrials)
    tmp1::Matrix{T}           # (latent_dim, ntrials)
    tmp2::Matrix{T}           # (latent_dim, ntrials)
    tmp3::Matrix{T}           # (latent_dim, ntrials)
    y::Array{T,3}             # (obs_dim, tsteps, ntrials)
    ux::Array{T,3}            # (ux_dim, tsteps, ntrials)
    uy::Array{T,3}            # (uy_dim, tsteps, ntrials)
    data_valid::Base.RefValue{Bool}  # true after first populate
end

function BatchedBuffers(
    ::Type{T},
    latent_dim::Int,
    obs_dim::Int,
    tsteps::Int,
    ntrials::Int;
    ux_dim::Int=0,
    uy_dim::Int=0,
) where {T<:Real}
    return BatchedBuffers{T}(
        zeros(T, latent_dim, tsteps, ntrials),  # x_mat
        zeros(T, latent_dim, tsteps, ntrials),  # grad_buf
        zeros(T, latent_dim, ntrials),          # dxt
        zeros(T, latent_dim, ntrials),          # dxt_next
        zeros(T, obs_dim, ntrials),             # dyt
        zeros(T, latent_dim, ntrials),          # tmp1
        zeros(T, latent_dim, ntrials),          # tmp2
        zeros(T, latent_dim, ntrials),          # tmp3
        zeros(T, obs_dim, tsteps, ntrials),     # y
        zeros(T, ux_dim, tsteps, ntrials),      # ux
        zeros(T, uy_dim, tsteps, ntrials),      # uy
        Ref(false),                             # data_valid
    )
end

"""
    SmoothWorkspace{T<:Real}

Pre-allocated workspace for the full LDS smoothing + EM pipeline, grouped by
concern:

- `btd`: block tridiagonal solver storage
- `consts`: Cholesky-derived model constants ([`SmoothConstants`](@ref))
- `opt`: Newton-smoother iterate / gradient / kernel scratch
- `reg`: regression-shaped M-step + aggregator buffers
- `elbo`: `Q_state!` / `Q_obs!` accumulators
- `agg`: TD sufficient-stats aggregator + shared-covariance storage
- `batched`: batched mean-pass buffers, or `nothing` (only `sws_pool[1]` of a
  multi-trial equal-length fit carries one)
"""
struct SmoothWorkspace{T<:Real}
    btd::BlockTridiagonalWorkspace{T}
    consts::SmoothConstants{T}
    opt::NewtonBuffers{T}
    reg::RegressionBuffers{T}
    elbo::ElboBuffers{T}
    agg::TDAggBuffers{T}
    batched::Union{Nothing,BatchedBuffers{T}}
end

"""
    SmoothWorkspace(::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int;
                    ux_dim=0, uy_dim=0, ntrials=1)

Construct a preallocated `SmoothWorkspace` for the full LDS EM pipeline.

- `ux_dim` is the dynamics-input dimension (`size(state_model.B, 2)`), used to
  size the regression buffers to fit `[A b B]`.
- `uy_dim` is the observation-input dimension (`size(obs_model.D, 2)`), used
  to size the regression buffers to fit `[C d D]`.
- `ntrials > 1` allocates the batched mean-pass buffers used by the
  equal-length cov-cache fast path (`batched` is `nothing` otherwise). Only
  `sws_pool[1]` at fit entry needs the real `ntrials`.

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
    batched = if ntrials > 1
        BatchedBuffers(T, latent_dim, obs_dim, tsteps, ntrials; ux_dim=ux_dim, uy_dim=uy_dim)
    else
        nothing
    end
    return SmoothWorkspace{T}(
        BlockTridiagonalWorkspace(T, latent_dim, tsteps),                                   # block tridiagonal solver storage (e-step)
        SmoothConstants(T, latent_dim, obs_dim),                                            # Cholesky-derived model constants (e-step)
        NewtonBuffers(T, latent_dim, obs_dim, tsteps),                                      # Newton solver buffers (e-step)
        RegressionBuffers(T, latent_dim, obs_dim; ux_dim=ux_dim, uy_dim=uy_dim),            # Buffers for m-step regression updates
        ElboBuffers(T, latent_dim, obs_dim),                                                # Buffers for Q_state! / Q_obs! ELBO terms
        TDAggBuffers(T, latent_dim, obs_dim, tsteps; ux_dim=ux_dim, uy_dim=uy_dim),         # Buffers for TD sufficient-statistics aggregator + shared smoothed-covariance storage
        batched,
    )
end

"""
    compute_smooth_constants!(cc::SmoothConstants, lds)
    compute_smooth_constants!(ws::SmoothWorkspace, lds)

Pre-compute and cache all Cholesky factors and derived terms that are constant
within a single `smooth!` call (i.e., depend only on model parameters, not on x).
Must be called once at the start of each `smooth!` invocation (and once per
regime after each SLDS M-step — see `refresh_slds_constants!`).

Dispatches on the observation model type:
- Gaussian: computes both state and observation model terms.
- Poisson: only computes state model terms (observation terms are x-dependent);
  `yt_given_xt` / `C_inv_R` are zeroed and `cR` is zero.

The `SmoothWorkspace` form forwards to the workspace's embedded
`SmoothConstants`.
"""
function compute_smooth_constants!(
    cc::SmoothConstants{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R

    #=
    Rewrap covariances as PDMats — each PDMat caches its own Cholesky
    factor internally and is consumed downstream via `cc.X_PD.chol.U`
    for triangular solves and `logdet(cc.X_PD)` for the normalizer.

    When `WT === T` (the hot path) `convert(Matrix{WT}, M)` returns `M`
    unchanged — no copy, no alloc. When the constants eltype differs
    (e.g. `ForwardDiff.Dual` for autodiff `loglikelihood`), constructing
    the PDMat directly with `WT`-typed factors avoids the
    `convert(::Type{PDMat{WT}}, ::PDMat{T})` fallback that requires a
    single-arg `Cholesky{WT}(::Cholesky{T})` method — present in
    Julia 1.12 but not Julia 1.10's stdlib `LinearAlgebra`.
    =#
    R_w = convert(Matrix{WT}, R)
    Q_w = convert(Matrix{WT}, Q)
    P0_w = convert(Matrix{WT}, P0)
    cc.R_PD = PDMat(Symmetrize!(R_w))
    cc.Q_PD = PDMat(Symmetrize!(Q_w))
    cc.P0_PD = PDMat(Symmetrize!(P0_w))
    Rchol = cc.R_PD.chol
    Qchol = cc.Q_PD.chol
    P0chol = cc.P0_PD.chol

    # tmp_RC = R^{-1} C
    copyto!(cc.tmp_RC, C)
    ldiv!(Rchol, cc.tmp_RC)
    copyto!(cc.C_inv_R, cc.tmp_RC')

    # tmp_QA = Q^{-1} A
    copyto!(cc.tmp_QA, A)
    ldiv!(Qchol, cc.tmp_QA)
    copyto!(cc.A_inv_Q, cc.tmp_QA')
    copyto!(cc.H_sub_entry, cc.tmp_QA)
    copyto!(cc.H_super_entry, cc.tmp_QA')

    # yt_given_xt = -C' * (R^{-1} C)
    mul!(cc.yt_given_xt, C', cc.tmp_RC)
    cc.yt_given_xt .*= -one(T)

    # xt_given_xt_1 = -Q^{-1}
    copyto!(cc.xt_given_xt_1, cc.I_mat)
    ldiv!(Qchol, cc.xt_given_xt_1)
    cc.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A' * (Q^{-1} A)
    mul!(cc.xt1_given_xt, A', cc.tmp_QA)
    cc.xt1_given_xt .*= -one(T)

    # x_t = -P0^{-1}
    copyto!(cc.x_t, cc.I_mat)
    ldiv!(P0chol, cc.x_t)
    cc.x_t .*= -one(T)

    # Log-likelihood normalizers (consumed by the likelihood kernels)
    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim
    cc.cP0 = -WT(0.5) * (WT(latent_dim) * log(WT(2π)) + logdet(cc.P0_PD))
    cc.cQ = -WT(0.5) * (WT(latent_dim) * log(WT(2π)) + logdet(cc.Q_PD))
    cc.cR = -WT(0.5) * (WT(obs_dim) * log(WT(2π)) + logdet(cc.R_PD))

    return nothing
end

function compute_smooth_constants!(
    cc::SmoothConstants{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0

    #=
    Wrap state-side covariances as PDMats (the Poisson emission has no
    covariance, so R_PD stays on its identity placeholder). See the
    Gaussian overload for the `convert` rationale: it's a no-op when
    `WT === T` and avoids a Julia 1.10 `Cholesky` convert-method gap
    when `WT !== T` (ForwardDiff path).
    =#
    Q_w = convert(Matrix{WT}, Q)
    P0_w = convert(Matrix{WT}, P0)
    cc.Q_PD = PDMat(Symmetrize!(Q_w))
    cc.P0_PD = PDMat(Symmetrize!(P0_w))
    Q_chol = cc.Q_PD.chol
    P0_chol = cc.P0_PD.chol

    # Gradient terms: A_inv_Q = (Q_chol \ A)'
    copyto!(cc.tmp_QA, A)
    ldiv!(Q_chol, cc.tmp_QA)
    copyto!(cc.A_inv_Q, cc.tmp_QA')

    # Hessian block templates for state model
    copyto!(cc.H_sub_entry, cc.tmp_QA)          # Q_chol \ A
    copyto!(cc.H_super_entry, cc.tmp_QA')       # (Q_chol \ A)'

    # xt_given_xt_1 = -(Q_chol \ I) = -Q^{-1}
    copyto!(cc.xt_given_xt_1, cc.I_mat)
    ldiv!(Q_chol, cc.xt_given_xt_1)
    cc.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A' * (Q_chol \ A)
    mul!(cc.xt1_given_xt, A', cc.tmp_QA)
    cc.xt1_given_xt .*= -one(T)

    # x_t = -(P0_chol \ I) = -P0^{-1}
    copyto!(cc.x_t, cc.I_mat)
    ldiv!(P0_chol, cc.x_t)
    cc.x_t .*= -one(T)

    # Emission-side templates are x-dependent for Poisson; zero the cached
    # Gaussian ones so no stale values survive an observation-model switch.
    fill!(cc.yt_given_xt, zero(WT))
    fill!(cc.C_inv_R, zero(WT))

    # Log-likelihood normalizers. No R term for Poisson observations.
    latent_dim = lds.latent_dim
    cc.cP0 = -WT(0.5) * (WT(latent_dim) * log(WT(2π)) + logdet(cc.P0_PD))
    cc.cQ = -WT(0.5) * (WT(latent_dim) * log(WT(2π)) + logdet(cc.Q_PD))
    cc.cR = zero(WT)

    return nothing
end

function compute_smooth_constants!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    return compute_smooth_constants!(ws.consts, lds)
end

"""
    _copy_smooth_constants!(dst::SmoothConstants, src::SmoothConstants)

Copy all fields populated by `compute_smooth_constants!` from `src` to `dst`.
Used by the equal-length multi-trial fast path to amortize the constants —
`_precompute_shared_cov!` runs `compute_smooth_constants!` once on the
designated workspace, and each per-task worker copies into its own
`SmoothConstants` instead of recomputing the Cholesky factors and derived
terms per trial. The PDMat wrappers are shared by reference (they are
immutable within an E-step); the buffer copies are pure `copyto!` over
fixed-size matrices and do not allocate.

Only the Gaussian-observation set of fields is copied; Poisson fits don't
go through the cov-cache fast path.
"""
function _copy_smooth_constants!(
    dst::SmoothConstants{T}, src::SmoothConstants{T}
) where {T<:Real}
    dst.R_PD = src.R_PD
    dst.Q_PD = src.Q_PD
    dst.P0_PD = src.P0_PD
    dst.cP0 = src.cP0
    dst.cQ = src.cQ
    dst.cR = src.cR
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

"""
    SLDSSmoothWorkspace{T}

Workspace for SLDS smoothing that matches the LDS backend shape:
- Owns a BlockTridiagonalWorkspace (H blocks + sparse + inverse scratch)
- Owns one [`SmoothConstants`](@ref) per SLDS component
- Owns a [`NewtonBuffers`](@ref) (`opt`) — the same iterate / gradient /
  kernel scratch layout as `SmoothWorkspace`, so kernels shared between the
  single-LDS and SLDS paths use one set of field paths
- `ll_tmp`: per-component log-likelihood scratch; the weighted accumulation
  across components needs a second `tsteps` buffer beside `opt.ll_vec`
"""
struct SLDSSmoothWorkspace{T<:Real}
    btd::BlockTridiagonalWorkspace{T}
    consts::Vector{SmoothConstants{T}}
    opt::NewtonBuffers{T}
    ll_tmp::Vector{T}   # per-component scratch (length tsteps)
end

function SLDSSmoothWorkspace(::Type{T}, slds::SLDS, tsteps::Int) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    K = length(slds.LDSs)

    ws = SLDSSmoothWorkspace{T}(
        BlockTridiagonalWorkspace(T, latent_dim, tsteps),
        [SmoothConstants(T, latent_dim, obs_dim) for _ in 1:K],
        NewtonBuffers(T, latent_dim, obs_dim, tsteps),
        zeros(T, tsteps),                # ll_tmp
    )

    # Cache constants once
    refresh_slds_constants!(ws, slds)

    return ws
end

"""
Refresh the per-regime constant caches after an M-step has updated the LDS parameters.
Must be called before the next E-step so that Cholesky factors, Hessian templates, etc.
reflect the current Q, R, A, P0.
"""
function refresh_slds_constants!(ws::SLDSSmoothWorkspace{T}, slds) where {T}
    for k in eachindex(slds.LDSs)
        compute_smooth_constants!(ws.consts[k], slds.LDSs[k])
    end
    return nothing
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
