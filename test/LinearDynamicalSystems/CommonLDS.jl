# Common test utilities for Linear Dynamical Systems
# Shared across GaussianLDS and PoissonLDS tests

# Common test types
const CHECKED_TYPES = [Float32, Float64] #, BigFloat]

"""
    test_type_preservation_for_model(create_model_fn, check_fields_fn, types=CHECKED_TYPES)

Generic test for type preservation across different numeric types.

# Arguments
- `create_model_fn(T)`: Function that creates a model with element type T
- `check_fields_fn(model, T)`: Function that performs @test assertions on model fields
- `types`: Collection of types to test
"""
function test_type_preservation_for_model(
    create_model_fn, check_fields_fn, types=CHECKED_TYPES
)
    for T in types
        model = create_model_fn(T)
        check_fields_fn(model, T)
    end
end

"""
    test_gradient_common(lds, x, y)

Test that analytical gradient matches numerical gradient for any LDS type.
Works for both GaussianLDS and PoissonLDS.
"""
function test_gradient_common(lds, x, y)
    for i in axes(y, 3)
        f = latents -> sum(StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i]))
        grad_numerical = ForwardDiff.gradient(f, x[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x[:, :, i])
        @test norm(grad_numerical - grad_analytical) < 1e-8
    end
end

"""
    test_hessian_common(lds, x, y)

Test that analytical Hessian matches numerical Hessian for any LDS type.
Works for both GaussianLDS and PoissonLDS.
"""
function test_hessian_common(lds, x, y)
    function log_likelihood(x::AbstractArray, lds, y::AbstractArray)
        return sum(StateSpaceDynamics.loglikelihood(x, lds, y))
    end

    for i in axes(y, 3)
        hess, main, super, sub = StateSpaceDynamics.Hessian(lds, y[:, 1:3, i], x[:, 1:3, i])
        @test size(hess) == (3 * lds.latent_dim, 3 * lds.latent_dim)
        @test size(main) == (3,)
        @test size(super) == (2,)
        @test size(sub) == (2,)

        obj = latents -> log_likelihood(latents, lds, y[:, 1:3, i])
        hess_numerical = ForwardDiff.hessian(obj, x[:, 1:3, i])
        @test norm(hess_numerical - hess) < 1e-8
    end
end

"""
    test_smooth_common(lds, x, y)

Test smoothing produces correct dimensions and gradients are near zero at the mode.
Works for both GaussianLDS and PoissonLDS.
"""
function test_smooth_common(lds, x, y)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, size(y, 2), size(y, 3))
    StateSpaceDynamics.smooth!(lds, tfs, y)

    n_trials = size(y, 3)
    n_tsteps = size(y, 2)

    x_smooth = tfs[1].x_smooth
    p_smooth = tfs[1].p_smooth
    p_smooth_tt1 = tfs[1].p_smooth_tt1

    @test size(x_smooth) == (lds.latent_dim, n_tsteps)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test size(p_smooth_tt1) == (lds.latent_dim, lds.latent_dim, n_tsteps)

    # Test that gradient is near zero at the smoothed estimate
    for i in axes(y, 3)
        f = latents -> sum(StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i]))
        grad_numerical = ForwardDiff.gradient(f, x_smooth[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x_smooth[:, :, i])
        @test norm(grad_numerical - grad_analytical) < 1e-7
    end
end

"""
    test_estep_common(lds, x, y)

Test E-step produces correct dimensions and valid marginal likelihood.
Works for both GaussianLDS and PoissonLDS.
"""
function test_estep_common(lds, x, y)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, size(y, 2), size(y, 3))
    ml_total = StateSpaceDynamics.estep!(lds, tfs, y)

    n_trials = size(y, 3)
    n_tsteps = size(y, 2)

    E_z, E_zz, E_zz_prev = tfs[1].E_z, tfs[1].E_zz, tfs[1].E_zz_prev
    x_smooth, p_smooth = tfs[1].x_smooth, tfs[1].p_smooth

    @test size(E_z) == (lds.latent_dim, n_tsteps)
    @test size(E_zz) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test size(E_zz_prev) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test size(x_smooth) == (lds.latent_dim, n_tsteps)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test isa(ml_total, Float64)
end

"""
    test_initial_state_parameter_updates_common(toy_fn, ntrials=1)

Test that initial state parameters (x0, P0) are updated correctly via M-step.
Works for both GaussianLDS and PoissonLDS.
"""
function test_initial_state_parameter_updates_common(toy_fn, ntrials=1)
    lds, x, y = toy_fn(ntrials, [true, true, false, false, false, false])

    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, size(y, 2), size(y, 3))
    ml_total = StateSpaceDynamics.estep!(lds, tfs, y)

    function obj(x0::AbstractVector, P0_sqrt::AbstractMatrix, lds)
        A, b, Q = lds.state_model.A, lds.state_model.b, lds.state_model.Q
        P0 = P0_sqrt * P0_sqrt'
        Q_val = 0.0
        for i in 1:ntrials
            E_z, E_zz, E_zz_prev = tfs[i].E_z, tfs[i].E_zz, tfs[i].E_zz_prev
            Q_val += StateSpaceDynamics.Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
        end
        return -Q_val
    end

    P0_sqrt = Matrix(cholesky(lds.state_model.P0).U)

    x0_opt = optimize(
        x0 -> obj(x0, P0_sqrt, lds),
        lds.state_model.x0,
        LBFGS(),
        Optim.Options(; g_abstol=1e-12),
    ).minimizer
    P0_opt = optimize(P0_ -> obj(x0_opt, P0_, lds), P0_sqrt, LBFGS()).minimizer

    StateSpaceDynamics.mstep!(lds, tfs, y)

    @test isapprox(lds.state_model.x0, x0_opt, atol=1e-6)
    @test isapprox(lds.state_model.P0, P0_opt * P0_opt', atol=1e-6)
end

"""
    test_state_model_parameter_updates_common(toy_fn, ntrials=1)

Test that state model parameters (A, b, Q) are updated correctly via M-step.
Works for both GaussianLDS and PoissonLDS.
"""
function test_state_model_parameter_updates_common(toy_fn, ntrials=1)
    lds, x, y = toy_fn(ntrials, [false, false, true, true, false, false])

    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, size(y, 2), size(y, 3))
    ml_total = StateSpaceDynamics.estep!(lds, tfs, y)

    function obj_state(AB::AbstractMatrix, Q_sqrt::AbstractMatrix, lds)
        D = size(AB, 1)
        A = AB[:, 1:D]
        b = AB[:, D + 1]
        Q = Q_sqrt * Q_sqrt'
        val = zero(eltype(Q))
        @views for k in 1:ntrials
            E_z, E_zz, E_zz_prev = tfs[k].E_z, tfs[k].E_zz, tfs[k].E_zz_prev
            val += StateSpaceDynamics.Q_state(
                A, b, Q, lds.state_model.P0, lds.state_model.x0, E_z, E_zz, E_zz_prev
            )
        end
        return -val
    end

    D = lds.latent_dim
    AB0 = hcat(lds.state_model.A, lds.state_model.b)
    Q_sqrt0 = Matrix(cholesky(lds.state_model.Q).U)

    AB_opt = optimize(AB -> obj_state(AB, Q_sqrt0, lds), AB0, LBFGS()).minimizer
    Q_opt_sqrt = optimize(Qs -> obj_state(AB_opt, Qs, lds), Q_sqrt0, LBFGS()).minimizer

    StateSpaceDynamics.mstep!(lds, tfs, y)

    @test isapprox(lds.state_model.A, AB_opt[:, 1:D], atol=1e-6, rtol=1e-6)
    @test isapprox(lds.state_model.b, AB_opt[:, D + 1], atol=1e-6, rtol=1e-6)
    @test isapprox(lds.state_model.Q, Q_opt_sqrt * Q_opt_sqrt', atol=1e-6, rtol=1e-6)
end

"""
    test_em_convergence_common(toy_fn, n_trials=1)

Test that EM algorithm produces monotonically increasing likelihood/ELBO.
Works for both GaussianLDS and PoissonLDS.
"""
function test_em_convergence_common(toy_fn, n_trials=1)
    lds, x, y = toy_fn(n_trials)
    objective, param_diff = fit!(lds, y; max_iter=100)
    # For GaussianLDS this is marginal likelihood, for PoissonLDS it's ELBO
    # Both should be monotonically increasing
    @test objective[end] > objective[1]
end
