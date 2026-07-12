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
`x`, `y` are vectors of per-trial matrices.
"""
function test_gradient_common(lds, x, y)
    for i in eachindex(y)
        f = latents -> sum(StateSpaceDynamics.joint_loglikelihood(lds, latents, y[i]))
        grad_numerical = ForwardDiff.gradient(f, x[i])
        ws = StateSpaceDynamics.SmoothWorkspace(
            Float64, lds.latent_dim, lds.obs_dim, size(y[i], 2)
        )
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        grad_analytical = copy(StateSpaceDynamics.gradient!(ws, lds, x[i], y[i]))
        @test norm(grad_numerical - grad_analytical) < 1e-8
    end
end

"""
    test_hessian_common(lds, x, y)

Test that analytical Hessian matches numerical Hessian for any LDS type.
"""
function test_hessian_common(lds, x, y)
    function log_likelihood(x::AbstractArray, lds, y::AbstractArray)
        return sum(StateSpaceDynamics.joint_loglikelihood(lds, x, y))
    end

    tsteps_test = 3
    ws = StateSpaceDynamics.SmoothWorkspace(
        Float64, lds.latent_dim, lds.obs_dim, tsteps_test
    )

    for i in eachindex(y)
        yi = y[i][:, 1:tsteps_test]
        xi = x[i][:, 1:tsteps_test]

        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        StateSpaceDynamics.hessian!(ws, lds, xi, yi)

        btd = ws.btd
        @test length(btd.H_diag) == tsteps_test
        @test length(btd.H_super) == tsteps_test - 1
        @test length(btd.H_sub) == tsteps_test - 1

        hess = block_tridgm(btd.H_diag, btd.H_super, btd.H_sub)
        obj = latents -> log_likelihood(latents, lds, yi)
        hess_numerical = ForwardDiff.hessian(obj, xi)
        @test norm(hess_numerical - hess) < 1e-8
    end
end

"""
    test_smooth_common(lds, x, y)

Test smoothing produces correct dimensions and gradients are near zero at the mode.
"""
function test_smooth_common(lds, x, y)
    tsteps_per_trial = [size(yt, 2) for yt in y]
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
    StateSpaceDynamics.smooth!(lds, tfs, y)

    n_tsteps = size(y[1], 2)
    x_smooth = tfs[1].x_smooth
    p_smooth = tfs[1].p_smooth
    p_smooth_tt1 = tfs[1].p_smooth_tt1

    @test size(x_smooth) == (lds.latent_dim, n_tsteps)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test size(p_smooth_tt1) == (lds.latent_dim, lds.latent_dim, n_tsteps)

    for i in eachindex(y)
        f = latents -> sum(StateSpaceDynamics.joint_loglikelihood(lds, latents, y[i]))
        grad_numerical = ForwardDiff.gradient(f, tfs[i].x_smooth)
        ws = StateSpaceDynamics.SmoothWorkspace(
            Float64, lds.latent_dim, lds.obs_dim, size(y[i], 2)
        )
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        grad_analytical = copy(StateSpaceDynamics.gradient!(ws, lds, tfs[i].x_smooth, y[i]))
        @test norm(grad_numerical - grad_analytical) < 1e-7
    end

    #=
    The allocating `smooth` convenience wrappers (single-trial + multi-trial)
    should reproduce the in-place `smooth!` result: same algorithm, fresh
    buffers. This is the only direct coverage of those public entry points.
    =#
    xs_multi, Ps_multi = StateSpaceDynamics.smooth(lds, y)
    @test length(xs_multi) == length(y)
    @test length(Ps_multi) == length(y)
    for i in eachindex(y)
        x_single, P_single = StateSpaceDynamics.smooth(lds, y[i])
        @test size(x_single) == size(tfs[i].x_smooth)
        @test size(P_single) == size(tfs[i].p_smooth)
        @test x_single ≈ tfs[i].x_smooth
        @test P_single ≈ tfs[i].p_smooth
        @test xs_multi[i] ≈ tfs[i].x_smooth
        @test Ps_multi[i] ≈ tfs[i].p_smooth
    end
end

"""
    test_estep_common(lds, x, y)

Test that smoothing + sufficient_statistics! populates the FilterSmooth
fields with the right shapes. Replaces the legacy `estep!(lds, tfs, y)`
convenience — production fit! uses the suf-based path; tests run the two
pieces explicitly.
"""
function test_estep_common(lds, x, y)
    tsteps_per_trial = [size(yt, 2) for yt in y]
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
    T_max = maximum(tsteps_per_trial)
    sws_pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, T_max) for
        _ in 1:Threads.maxthreadid()
    ]
    StateSpaceDynamics.smooth!(lds, tfs, y, sws_pool)
    StateSpaceDynamics.sufficient_statistics!(tfs)

    n_tsteps = size(y[1], 2)

    E_z, E_zz, E_zz_prev = tfs[1].E_z, tfs[1].E_zz, tfs[1].E_zz_prev
    x_smooth, p_smooth = tfs[1].x_smooth, tfs[1].p_smooth

    @test size(E_z) == (lds.latent_dim, n_tsteps)
    @test size(E_zz) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test size(E_zz_prev) == (lds.latent_dim, lds.latent_dim, n_tsteps)
    @test size(x_smooth) == (lds.latent_dim, n_tsteps)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, n_tsteps)
end

"""
    test_initial_state_parameter_updates_common(toy_fn, ntrials=1)

Test that initial state parameters (x0, P0) are updated correctly via M-step.
"""
function test_initial_state_parameter_updates_common(toy_fn, ntrials=1)
    lds, x, y = toy_fn(ntrials, [true, true, false, false, false, false])
    tsteps_per_trial = [size(yt, 2) for yt in y]
    tsteps = tsteps_per_trial[1]

    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
    ws = StateSpaceDynamics.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, tsteps)

    # Smooth + populate E_z/E_zz/E_zz_prev for the reference Q_state objective.
    sws_pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, tsteps) for
        _ in 1:Threads.maxthreadid()
    ]
    StateSpaceDynamics.smooth!(lds, tfs, y, sws_pool)
    StateSpaceDynamics.sufficient_statistics!(tfs)

    x0_orig = copy(lds.state_model.x0)
    P0_orig = copy(lds.state_model.P0)

    function obj(x0::AbstractVector, P0_sqrt::AbstractMatrix)
        lds.state_model.x0 .= x0
        lds.state_model.P0 .= P0_sqrt * P0_sqrt'
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        Q_val = 0.0
        for i in 1:ntrials
            Q_val += StateSpaceDynamics.Q_state!(
                ws, lds, tfs[i].E_z, tfs[i].E_zz, tfs[i].E_zz_prev
            )
        end
        return -Q_val
    end

    P0_sqrt = Matrix(cholesky(P0_orig).U)

    x0_opt =
        optimize(
            x0 -> obj(x0, P0_sqrt), copy(x0_orig), LBFGS(), Optim.Options(; g_abstol=1e-12)
        ).minimizer
    P0_opt = optimize(P0_ -> obj(x0_opt, P0_), P0_sqrt, LBFGS()).minimizer

    lds.state_model.x0 .= x0_orig
    lds.state_model.P0 .= P0_orig

    # M-step via the suf-based path: aggregate sufficient statistics from
    # tfs.x_smooth/p_smooth/p_smooth_tt1, then run mstep!(lds, suf, ws).
    suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
        Float64, lds, tsteps_per_trial
    )
    ux_seq = [zeros(Float64, 0, size(yt, 2)) for yt in y]
    uy_seq = [zeros(Float64, 0, size(yt, 2)) for yt in y]
    StateSpaceDynamics._td_init_const_blocks!(ws, lds, tsteps_per_trial, y, ux_seq, uy_seq)
    StateSpaceDynamics._aggregate_td_suff_stats!(suf, tfs, lds, ux_seq, uy_seq, y, ws)
    if lds.obs_model isa StateSpaceDynamics.GaussianObservationModel
        StateSpaceDynamics.mstep!(lds, suf, ws)
    else
        StateSpaceDynamics.update_initial_state_mean!(lds, suf)
        StateSpaceDynamics.update_initial_state_covariance!(lds, suf, ws)
        StateSpaceDynamics.update_A_b!(lds, suf, ws)
        StateSpaceDynamics.update_Q!(lds, suf, ws)
        StateSpaceDynamics.update_observation_model!(lds, tfs, y, [ws])
    end

    @test isapprox(lds.state_model.x0, x0_opt, atol=1e-6)
    @test isapprox(lds.state_model.P0, P0_opt * P0_opt', atol=1e-6)
end

"""
    test_state_model_parameter_updates_common(toy_fn, ntrials=1)

Test that state model parameters (A, b, Q) are updated correctly via M-step.
"""
function test_state_model_parameter_updates_common(toy_fn, ntrials=1)
    lds, x, y = toy_fn(ntrials, [false, false, true, true, false, false])
    tsteps_per_trial = [size(yt, 2) for yt in y]
    tsteps = tsteps_per_trial[1]

    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
    ws = StateSpaceDynamics.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, tsteps)
    sws_pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, tsteps) for
        _ in 1:Threads.maxthreadid()
    ]
    StateSpaceDynamics.smooth!(lds, tfs, y, sws_pool)
    StateSpaceDynamics.sufficient_statistics!(tfs)

    A_orig = copy(lds.state_model.A)
    b_orig = copy(lds.state_model.b)
    Q_orig = copy(lds.state_model.Q)

    function obj_state(AB::AbstractMatrix, Q_sqrt::AbstractMatrix)
        D = size(AB, 1)
        lds.state_model.A .= AB[:, 1:D]
        lds.state_model.b .= AB[:, D + 1]
        lds.state_model.Q .= Q_sqrt * Q_sqrt'
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        val = 0.0
        @views for k in 1:ntrials
            val += StateSpaceDynamics.Q_state!(
                ws, lds, tfs[k].E_z, tfs[k].E_zz, tfs[k].E_zz_prev
            )
        end
        return -val
    end

    D = lds.latent_dim
    AB0 = hcat(A_orig, b_orig)
    Q_sqrt0 = Matrix(cholesky(Q_orig).U)

    AB_opt = optimize(AB -> obj_state(AB, Q_sqrt0), AB0, LBFGS()).minimizer
    Q_opt_sqrt = optimize(Qs -> obj_state(AB_opt, Qs), Q_sqrt0, LBFGS()).minimizer

    lds.state_model.A .= A_orig
    lds.state_model.b .= b_orig
    lds.state_model.Q .= Q_orig

    suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
        Float64, lds, tsteps_per_trial
    )
    ux_seq = [zeros(Float64, 0, size(yt, 2)) for yt in y]
    uy_seq = [zeros(Float64, 0, size(yt, 2)) for yt in y]
    StateSpaceDynamics._td_init_const_blocks!(ws, lds, tsteps_per_trial, y, ux_seq, uy_seq)
    StateSpaceDynamics._aggregate_td_suff_stats!(suf, tfs, lds, ux_seq, uy_seq, y, ws)
    if lds.obs_model isa StateSpaceDynamics.GaussianObservationModel
        StateSpaceDynamics.mstep!(lds, suf, ws)
    else
        StateSpaceDynamics.update_initial_state_mean!(lds, suf)
        StateSpaceDynamics.update_initial_state_covariance!(lds, suf, ws)
        StateSpaceDynamics.update_A_b!(lds, suf, ws)
        StateSpaceDynamics.update_Q!(lds, suf, ws)
        StateSpaceDynamics.update_observation_model!(lds, tfs, y, [ws])
    end

    @test isapprox(lds.state_model.A, AB_opt[:, 1:D], atol=1e-6, rtol=1e-6)
    @test isapprox(lds.state_model.b, AB_opt[:, D + 1], atol=1e-6, rtol=1e-6)
    @test isapprox(lds.state_model.Q, Q_opt_sqrt * Q_opt_sqrt', atol=1e-6, rtol=1e-6)
end

"""
    test_em_convergence_common(toy_fn, n_trials=1)

Test that EM algorithm produces monotonically increasing likelihood/ELBO.
"""
function test_em_convergence_common(toy_fn, n_trials=1)
    # Seed via StableRNGs so the sampled dataset is reproducible
    Random.seed!(Random.default_rng(), rand(StableRNG(20260510), UInt))
    lds, x, y = toy_fn(n_trials)
    objective = fit!(lds, y; max_iter=100)
    @test objective[end] > objective[1]
end
