# =============================================================================
# Conjugate priors used across LDS-family models.
#
# Currently:
#   * `IWPrior`  — inverse-Wishart prior on a covariance matrix (Q, P0, R, ...).
#   * `MNPrior`  — matrix-normal prior on a regression coefficient matrix
#                  ([A B], [C D], ...). Pair with an `IWPrior` on the same
#                  regression to obtain the full MNIW conjugate prior on
#                  (W, Σ).
#
# Each prior ships with a small `*_map` helper that returns the closed-form
# MAP update so M-steps stay one-liner-clean.
# =============================================================================

"""
    IWPrior{T<:Real, M<:AbstractMatrix}

Inverse-Wishart prior for a covariance matrix Σ ~ IW(Ψ, ν), with density
p(Σ) ∝ |Σ|^{-(ν + d + 1)/2} exp(-½ tr(Ψ Σ^{-1})) for d = size(Σ,1).

# Fields
- `Ψ::M`: Scale matrix (d×d, SPD).
- `ν::T`: Degrees of freedom (must satisfy `ν > d + 1` for a proper mode).

# Notes
- The MAP update for a posterior IW(Ψ + S, ν + n) is `(Ψ + S) / (ν + n + d + 1)`.
"""
Base.@kwdef struct IWPrior{T<:Real,M<:AbstractMatrix}
    Ψ::M
    ν::T
end

# helpers for new priors on cov matrices
@inline function iw_map(
    Ψ::AbstractMatrix{T}, ν::T, S::AbstractMatrix{T}, n::T, d::Int
) where {T}
    return (Ψ .+ S) ./ (ν + n + d + one(T))
end

# TODO: this should use PD Mats
@inline function iw_logprior_term(Σ::AbstractMatrix{T}, prior::IWPrior{T}) where {T}
    D = size(Σ, 1)
    Ψ, ν = prior.Ψ, prior.ν
    # log|Σ| via Cholesky
    F = cholesky(Symmetric(Σ))
    logdetΣ = 2sum(log, diag(F.U))
    # tr(Ψ Σ^{-1}) via triangular solves
    X = F \ Ψ                 # solves Σ * X = Ψ
    return -T(0.5) * ((ν + D + one(T)) * logdetΣ + tr(X))
end

"""
    MNPrior{T<:Real, M<:AbstractMatrix}

Matrix-normal prior on a regression coefficient matrix W (size `k × p`)
appearing in a linear regression `Y = W X + ε` with row-noise covariance Σ:

```math
W | Σ ~ MN(M₀, Σ ⊗ Λ⁻¹)
```

equivalently `vec(W) ~ N(vec(M₀), Λ⁻¹ ⊗ Σ)`. Σ is the regression's row
covariance (paired with `IWPrior` when both halves of an MNIW prior are
desired); Λ is the column precision and is the only piece that enters the MAP
update for W.

# Fields
- `M₀::M`: prior mean (`k × p`, same shape as W). Use a zero matrix for plain
    ridge; an identity-like matrix for shrinkage toward a random walk on `A`.
- `Λ::M`: column precision (`p × p`, SPD).

# Notes
- The MAP update for W given the regression sufficient statistics `XX = X Xᵀ`
    and `XY = X Yᵀ` is
    ```math
    W = (XYᵀ + M₀ Λ) (XX + Λ)⁻¹
    ```
    Reduces to OLS when `Λ = 0`, and to ordinary ridge regression when
    `M₀ = 0`.
- Mathematically half of an MNIW prior; combine with an `IWPrior` on the same
    regression to recover the full conjugate prior on `(W, Σ)`.
"""
Base.@kwdef struct MNPrior{T<:Real,M<:AbstractMatrix{T}}
    M₀::M
    Λ::M
end

"""
    mn_map(XX, XY, prior) -> Matrix

MAP estimate for the regression coefficient `W` under an `MNPrior`:
returns `W = (XYᵀ + M₀ Λ)(XX + Λ)⁻¹`. Falls back to OLS (`W = XYᵀ XX⁻¹`)
when `prior === nothing`.

`XX` may be a plain `AbstractMatrix` or a `PDMat`; the latter is preferred so
the cached Cholesky of `XX + Λ` is reused (PDMats handles the addition).
"""
@inline function mn_map(
    XX::AbstractMatrix{T}, XY::AbstractMatrix{T}, prior::MNPrior{T}
) where {T}
    return transpose((XX + prior.Λ) \ (XY + prior.Λ * prior.M₀'))
end

@inline function mn_map(XX::AbstractMatrix{T}, XY::AbstractMatrix{T}, ::Nothing) where {T}
    return transpose(XX \ XY)
end

"""
    mn_logprior_term(W, Σ, prior) -> Real

W-dependent part of the matrix-normal log prior `log p(W | Σ)` evaluated at the
current `(W, Σ)`. Drops Σ- and Λ-only constants that the IW prior + the `mn_map`
M-step already cover, leaving the quadratic term

    -½ tr(Σ^{-1} (W - M₀) Λ (W - M₀)')

This is the contribution the ELBO needs to display the true MAP objective when
an `MNPrior` is set on a regression coefficient (e.g. `[A b B]` or `[C d D]`).
Without it, EM is still maximizing the right thing internally, but the displayed
ELBO can decrease under strong shrinkage — see git log for the fix context.
"""
@inline function mn_logprior_term(
    W::AbstractMatrix{T}, Σ::AbstractMatrix{T}, prior::MNPrior{T}
) where {T<:Real}
    Wm = W .- prior.M₀
    # tr(Σ^{-1} Wm Λ Wm') = tr(Λ Wm' Σ^{-1} Wm); pick the cheaper order based
    # on shape, but both are k × k after the trace, so just go left-to-right.
    M = Wm * prior.Λ * Wm'
    F = cholesky(Symmetric(Σ))
    return -T(0.5) * tr(F \ M)
end

@inline mn_logprior_term(W::AbstractMatrix, ::AbstractMatrix, ::Nothing) = zero(eltype(W))
