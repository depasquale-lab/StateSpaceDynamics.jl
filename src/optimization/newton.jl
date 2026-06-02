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
            # α == 0 means the line search found no monotone finite step (e.g. a
            # non-ascent/descent direction or a NaN/Inf objective region). The
            # gradient norm is still ≥ tol here (otherwise we'd have returned at
            # the top of the loop), so this is a stall.
            if iszero(α)
                return false
            end
            if abs(ϕ_new - ϕ_prev) < tol || α*norm(p) < tol
                return true
            end
            ϕ_prev = ϕ_new
        end
    end
    return false
end
