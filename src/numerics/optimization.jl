abstract type AbstractLineSearch end

Base.@kwdef struct BackTrackingLS{T} <: AbstractLineSearch
    c1::T = 1e-4
    ρ_hi::T = 0.5
    ρ_lo::T = 0.1
    max_iters::Int = 25
    max_halvings::Int = 50
    order::Int = 3
end

# Armijo check, written so we can handle max/min with Val
@inline function armijo_ok(::Val{:max}, ϕ, ϕ0, α, dϕ0, c1)
    return ϕ >= ϕ0 + c1 * α * dϕ0
end
@inline function armijo_ok(::Val{:min}, ϕ, ϕ0, α, dϕ0, c1)
    return ϕ <= ϕ0 + c1 * α * dϕ0
end

# Strict-improvement check in the sense direction. Phase 1 halves until
# `_improves` holds, guaranteeing every later step is monotone vs ϕ0.
@inline _improves(::Val{:max}, ϕ, ϕ0) = isfinite(ϕ) && ϕ > ϕ0
@inline _improves(::Val{:min}, ϕ, ϕ0) = isfinite(ϕ) && ϕ < ϕ0

#= Stationary point of the phase-2 cubic to step toward. The interpolant
c(α) = aα³ + bα² + dϕ0·α + ϕ0 has two stationary points (-b ± √disc)/(3a);
c''(α) = ±2√disc, so the +root is the local MINIMIZER and the -root the local
MAXIMIZER.
=#
@inline _cubic_step(::Val{:max}, a, b, disc) = (-b - sqrt(disc)) / (3 * a)
@inline _cubic_step(::Val{:min}, a, b, disc) = (-b + sqrt(disc)) / (3 * a)

"""
    backtracking!(sense, ls, x, p, ϕ!, ϕ0, dϕ0)

In-place backtracking along direction `p` from current `x`.
- `sense = Val(:max)` for maximizing ϕ; `Val(:min)` for minimizing.
- `ϕ!()` returns ϕ(x) using the current `x` (allocation-free).
Returns `(α, ϕ_new)`.

Phase 1 halves α until ϕ is finite *and* monotone vs ϕ0 — without the
monotone requirement, a finite-but-worse foothold can let phase 2's cubic
interpolation extrapolate back into `exp(...)` overflow (Poisson Newton
from a far-from-optimum start), produce NaN, and then poison the outer
Newton iterate. If phase 2 ever lands at a non-finite ϕ or non-finite
trial α, fall back to the best monotone step seen so far (`α_best`,
`ϕ_best`), which phase 1 guarantees exists.
"""
function backtracking!(
    sense::Val,
    ls::BackTrackingLS{T},
    x::AbstractArray{T},
    p::AbstractArray{T},
    ϕ!::F,
    ϕ0::T,
    dϕ0::T,
) where {T<:Real,F}
    @assert ls.order == 2 || ls.order == 3

    α1 = one(T)
    α2 = one(T)

    @. x = x + α2 * p
    ϕx0 = ϕ0
    ϕx1 = ϕ!()

    # Phase 1: halve until ϕ is finite *and* monotone vs ϕ0.
    h = 0
    while h < ls.max_halvings && !_improves(sense, ϕx1, ϕ0)
        h += 1
        @. x = x - α2 * p
        α1 = α2
        α2 *= T(0.5)
        @. x = x + α2 * p
        ϕx1 = ϕ!()
    end

    # No monotone finite step found — revert to x_start, no progress.
    if !_improves(sense, ϕx1, ϕ0)
        @. x = x - α2 * p
        return zero(T), ϕ0
    end

    # Best monotone step seen so far; fall back here on phase-2 failure.
    α_best = α2
    ϕ_best = ϕx1

    # Phase 2: cubic / quadratic interpolation.
    for k in 1:(ls.max_iters)
        if armijo_ok(sense, ϕx1, ϕ0, α2, dϕ0, ls.c1)
            return α2, ϕx1
        end

        αtmp = α2
        if ls.order == 2 || k == 1
            denom = (ϕx1 - ϕ0 - dϕ0 * α2)
            αtmp = -(dϕ0 * α2 * α2) / (2 * denom)
        else
            div = one(T) / (α1 * α1 * α2 * α2 * (α2 - α1))
            a = (α1 * α1 * (ϕx1 - ϕ0 - dϕ0 * α2) - α2 * α2 * (ϕx0 - ϕ0 - dϕ0 * α1)) * div
            b = (-α1^3 * (ϕx1 - ϕ0 - dϕ0 * α2) + α2^3 * (ϕx0 - ϕ0 - dϕ0 * α1)) * div
            if abs(a) < eps(T)
                αtmp = -dϕ0 / (2 * b)
            else
                disc = max(b * b - 3 * a * dϕ0, zero(T))
                αtmp = _cubic_step(sense, a, b, disc)
            end
        end

        αtmp = min(αtmp, α2 * ls.ρ_hi)
        αtmp = max(αtmp, α2 * ls.ρ_lo)

        if !isfinite(αtmp)
            @. x = x - α2 * p
            @. x = x + α_best * p
            return α_best, ϕ_best
        end

        @. x = x - α2 * p
        α1 = α2
        α2 = αtmp
        @. x = x + α2 * p

        ϕx0, ϕx1 = ϕx1, ϕ!()

        if !isfinite(ϕx1)
            @. x = x - α2 * p
            @. x = x + α_best * p
            return α_best, ϕ_best
        end

        if _improves(sense, ϕx1, ϕ_best)
            α_best = α2
            ϕ_best = ϕx1
        end
    end

    # Exhausted max_iters without satisfying Armijo: return the best monotone
    # step seen, not the last trial (which may be strictly worse).
    @. x = x - α2 * p
    @. x = x + α_best * p
    return α_best, ϕ_best
end

"""
    newton_smooth!(x, compute_grad!, build_hess!, solve_dir!, ϕ!, linesearch; ...)

- `x` is D×T (or vector view)
- `compute_grad!(g, x)` writes gradient into preallocated `g`
- `build_hess!(x)` fills workspace Hessian blocks
- `solve_dir!(p, g)` solves for Newton direction into preallocated `p`
- `ϕ!()` evaluates objective at current `x`
"""
function newton_smooth!(
    sense::Val,
    x,
    g,
    p,
    compute_grad!,
    build_hess!,
    solve_dir!,
    ϕ!,
    ls::Union{Nothing,AbstractLineSearch};
    max_iter::Int=20,
    tol=1e-6,
)
    ϕ_prev = -Inf
    for _ in 1:max_iter
        compute_grad!(g, x)
        gn = norm(g)
        if gn < tol
            return true
        end

        build_hess!(x)
        solve_dir!(p, g)  # p is Newton direction

        # step selection
        if ls === nothing
            @. x = x + p
            ϕ_prev = ϕ!()
        else
            ϕ0 = ϕ!()
            dϕ0 = dot(vec(g), vec(p))  # > 0 for ascent; < 0 for descent
            α, ϕ_new = backtracking!(sense, ls, x, p, ϕ!, ϕ0, dϕ0)
            #=
            α == 0 means the line search found no monotone finite step (e.g. a
            non-ascent/descent direction or a NaN/Inf objective region). The
            gradient norm is still ≥ tol here (otherwise we'd have returned at
            the top of the loop), so this is a stall.
            =#
            if iszero(α)
                return false
            end
            if abs(ϕ_new - ϕ_prev) < tol || α * norm(p) < tol
                return true
            end
            ϕ_prev = ϕ_new
        end
    end
    return false
end
