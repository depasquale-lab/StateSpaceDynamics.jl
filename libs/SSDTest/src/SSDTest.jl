"""
    SSDTest

Shared assertion helpers for the `docs/examples/` tutorials. Tutorials embed
calls into a `#src` section at the bottom (stripped from the rendered docs,
kept in the raw `.jl` so the test runner sees them). Pattern lifted from
HiddenMarkovModels.jl's `HMMTest` sub-package.
"""
module SSDTest

using LinearAlgebra
using Statistics
using StateSpaceDynamics
using Test

export test_em_monotone, test_em_improves, test_smooth_improves, test_lds_dimensions

"""
    test_em_monotone(elbos; tol=1e-6)

Assert the ELBO trajectory returned by [`fit!`](@ref) is non-decreasing
step-by-step (modulo a `tol` tolerance). Suitable for Gaussian LDS, where
EM is exactly monotone. For Laplace / variational EM use
[`test_em_improves`](@ref) instead — there the ELBO can dip locally even
though it improves overall.
"""
function test_em_monotone(elbos; tol::Real=1e-6)
    @testset "EM ELBO monotone" begin
        @test length(elbos) >= 1
        if length(elbos) > 1
            @test all(>=(-tol), diff(elbos))
            @test elbos[end] >= elbos[1] - tol
        end
    end
    return nothing
end

"""
    test_em_improves(elbos; tol=1e-6)

Assert the ELBO trajectory ends no worse than where it started. Use this
for Laplace-EM (PoissonLDS) and variational EM (SLDS) where the inner
approximation can cause small downward steps even on a well-behaved fit.
"""
function test_em_improves(elbos; tol::Real=1e-6)
    @testset "EM ELBO improves overall" begin
        @test length(elbos) >= 1
        if length(elbos) > 1
            @test elbos[end] >= elbos[1] - tol
        end
    end
    return nothing
end

"""
    test_smooth_improves(x_true, x_pre, x_post)

Assert that the smoothed-state estimate after EM is closer to the true
latents than the pre-EM estimate, after solving for the best linear map
between the two (latent coordinates are identifiable only up to invertible
change-of-basis). `x_true`, `x_pre`, `x_post` are each `D × T` matrices.
"""
function test_smooth_improves(
    x_true::AbstractMatrix, x_pre::AbstractMatrix, x_post::AbstractMatrix
)
    @testset "Smoothing improves with EM" begin
        @test size(x_true) == size(x_pre) == size(x_post)
        err_pre = _aligned_residual(x_true, x_pre)
        err_post = _aligned_residual(x_true, x_post)
        @test err_post <= err_pre
    end
    return nothing
end

function _aligned_residual(x_true::AbstractMatrix, x_est::AbstractMatrix)
    T_map = x_true / x_est
    return norm(x_true - T_map * x_est) / sqrt(length(x_true))
end

"""
    test_lds_dimensions(lds; latent_dim, obs_dim)

Sanity-check that a [`LinearDynamicalSystem`](@ref)'s state and observation
model dimensions match the expected sizes. Catches regressions where a
constructor silently accepts mis-shaped parameters.
"""
function test_lds_dimensions(lds; latent_dim::Int, obs_dim::Int)
    @testset "LDS dimensions" begin
        @test lds.latent_dim == latent_dim
        @test lds.obs_dim == obs_dim
        @test size(lds.state_model.A) == (latent_dim, latent_dim)
        @test size(lds.state_model.Q) == (latent_dim, latent_dim)
        @test size(lds.obs_model.C) == (obs_dim, latent_dim)
    end
    return nothing
end

end # module
