# Unit tests for the low-level optimization primitives in
# `src/optimization/{linesearch,newton}.jl`. These exercise code paths the
# higher-level LDS/Poisson smoothing tests do not reach:
#   - the `Val(:min)` sense of `backtracking!` (the smoothers only use `:max`),
#   - the "return the best monotone step on max_iters exhaustion" branch,
#   - the no-line-search (`ls === nothing`) Newton step,
#   - the non-converged (`return false`) exit of `newton_smooth!`.

const SSDopt = StateSpaceDynamics

function test_backtracking_min_sense_decreases()
    @testset "backtracking! :min decreases ϕ and advances x by α·p" begin
        # ϕ(s) = s², minimize from s = 2 along the descent direction p = -1.
        x = [2.0]
        ϕ!() = x[1]^2
        ls = SSDopt.BackTrackingLS{Float64}()
        ϕ0 = ϕ!()                     # 4.0
        g = [2 * x[1]]                # ∇ϕ = 2s
        p = [-1.0]                    # descent direction
        dϕ0 = g[1] * p[1]             # < 0 for a descent step

        α, ϕ_new = SSDopt.backtracking!(Val(:min), ls, x, p, ϕ!, ϕ0, dϕ0)

        @test α > 0
        @test ϕ_new < ϕ0              # Armijo (:min) accepted a strict decrease
        @test x ≈ [2.0 + α * p[1]]    # x mutated in place to x_start + α·p
        @test ϕ_new ≈ ϕ!()            # returned value matches ϕ at the returned x
    end
    return nothing
end

function test_backtracking_returns_best_on_exhaustion()
    @testset "backtracking! returns best monotone step when max_iters exhausts" begin
        #=
        Force Armijo to never accept (huge c1) and cap phase 2 at one iter, so
        the loop exhausts and must fall back to the best step seen — which is
        the phase-1 step (α=1, ϕ=1), strictly better than the lone phase-2
        trial it would otherwise have returned.
        =#
        x = [2.0]
        ϕ!() = x[1]^2
        ls = SSDopt.BackTrackingLS{Float64}(; c1=1e6, max_iters=1)
        ϕ0 = ϕ!()                     # 4.0
        p = [-1.0]
        dϕ0 = (2 * x[1]) * p[1]       # = -4.0

        α, ϕ_new = SSDopt.backtracking!(Val(:min), ls, x, p, ϕ!, ϕ0, dϕ0)

        @test α ≈ 1.0                 # the best (phase-1) step, not the last trial
        @test ϕ_new ≈ 1.0
        @test ϕ_new < ϕ0              # still a genuine improvement
        @test x ≈ [1.0]               # x restored to x_start + α_best·p
        @test ϕ_new ≈ ϕ!()            # x and returned ϕ are consistent
    end
    return nothing
end

# Shared minimal Newton setup: minimize ½(x-t)' H (x-t) with H SPD. The exact
# Newton direction solves H p = -g, landing on the target in a single step.
function _newton_quadratic_problem()
    H = [2.0 0.3; 0.3 1.0]
    target = [1.0, -1.0]
    x = [5.0, 5.0]
    g = similar(x)
    p = similar(x)
    compute_grad!(gv, xv) = (gv .= H * (xv .- target))
    build_hess!(_xv) = nothing
    solve_dir!(pv, gv) = (pv .= -(H \ gv))
    ϕ!() = 0.5 * dot(x .- target, H * (x .- target))
    return (; H, target, x, g, p, compute_grad!, build_hess!, solve_dir!, ϕ!)
end

function test_newton_smooth_no_linesearch_converges()
    @testset "newton_smooth! (ls = nothing) converges via gradient norm" begin
        pr = _newton_quadratic_problem()
        converged = SSDopt.newton_smooth!(
            Val(:min),
            pr.x,
            pr.g,
            pr.p,
            pr.compute_grad!,
            pr.build_hess!,
            pr.solve_dir!,
            pr.ϕ!,
            nothing;
            max_iter=10,
            tol=1e-10,
        )
        @test converged                 # gradient hit the tolerance
        @test pr.x ≈ pr.target          # the no-line-search step landed on the optimum
    end
    return nothing
end

function test_newton_smooth_returns_false_on_linesearch_stall()
    @testset "newton_smooth! returns false when the line search stalls (α = 0)" begin
        # Feed an *ascent* direction to a minimization: the line search can never
        # find a monotone (decreasing) step, so `backtracking!` returns α = 0.
        x = [1.0]
        ϕ!() = x[1]^2
        g = [0.0]
        p = [0.0]
        compute_grad!(gv, xv) = (gv .= 2 .* xv)         # ∇ϕ = 2s, large at s = 1
        build_hess!(_xv) = nothing
        solve_dir!(pv, gv) = (pv .= gv)                  # ascent direction (wrong sign)
        ls = SSDopt.BackTrackingLS{Float64}()

        converged = SSDopt.newton_smooth!(
            Val(:min),
            x,
            g,
            p,
            compute_grad!,
            build_hess!,
            solve_dir!,
            ϕ!,
            ls;
            max_iter=10,
            tol=1e-8,
        )
        @test converged == false        # stall is not convergence
        @test x ≈ [1.0]                 # no step was taken
    end
    return nothing
end

function test_newton_smooth_returns_false_on_max_iter()
    @testset "newton_smooth! returns false when max_iter is exhausted" begin
        #=
        Convergence is only detected at the top of the *next* iteration, so a
        single allowed iteration steps onto the target yet still reports
        non-convergence
        =#
        pr = _newton_quadratic_problem()
        converged = SSDopt.newton_smooth!(
            Val(:min),
            pr.x,
            pr.g,
            pr.p,
            pr.compute_grad!,
            pr.build_hess!,
            pr.solve_dir!,
            pr.ϕ!,
            nothing;
            max_iter=1,
            tol=1e-10,
        )
        @test converged == false        # exhausted max_iter without a top-of-loop check
        @test pr.x ≈ pr.target          # the single step still moved x onto the optimum
    end
    return nothing
end
