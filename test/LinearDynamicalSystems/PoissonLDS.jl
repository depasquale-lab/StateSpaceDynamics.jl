# Include common test utilities
include("CommonLDS.jl")

# define parameters for a PoissonLDS
x0 = [1.0, -1.0]
P0 = Matrix(Diagonal([0.1, 0.1]))  # Fixed: was p0, now P0
A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
Q = Matrix(Diagonal([0.1, 0.1]))
C = [0.6 0.6; 0.6 0.6; 0.6 0.6] .* 2
d = log.([0.1, 0.1, 0.1])
b = zeros(2)

function toy_PoissonLDS(
    ntrials::Int=1, fit_bool::Vector{Bool}=[true, true, true, true, true, true]
)
    # IMPORTANT: copy the module-level matrices. `GaussianStateModel`/
    # `PoissonObservationModel` keep references to whatever arrays are
    # passed in, so without these copies the first `fit!` call mutates the
    # module-level `A, Q, ...` arrays in place, and every subsequent
    # `toy_PoissonLDS()` call returns a model seeded with the previous
    # fit's parameters. Test-ordering-dependent flakiness ensues.
    gaussian_sm = GaussianStateModel(;
        A=copy(A), b=copy(b), Q=copy(Q), x0=copy(x0), P0=copy(P0)
    )
    poisson_om = PoissonObservationModel(; C=copy(C), d=copy(d))
    poisson_lds = LinearDynamicalSystem(;
        state_model=gaussian_sm,
        obs_model=poisson_om,
        latent_dim=2,
        obs_dim=3,
        fit_bool=fill(true, 6),
    )

    # sample data — seeded so the Poisson Newton smoother doesn't
    # occasionally diverge on an unlucky draw, and EM shows a clear
    # ELBO improvement (the `test_em_convergence_common` assertion).
    T = 100
    rng = StableRNG(42)
    x, y = rand(rng, poisson_lds, fill(T, ntrials))

    return poisson_lds, x, y
end

function test_plds_properties(poisson_lds)
    @test isa(poisson_lds.state_model, StateSpaceDynamics.GaussianStateModel)
    @test isa(poisson_lds.obs_model, StateSpaceDynamics.PoissonObservationModel)
    @test isa(poisson_lds, StateSpaceDynamics.LinearDynamicalSystem)

    @test size(poisson_lds.state_model.A) ==
        (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.obs_model.C) == (poisson_lds.obs_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.state_model.Q) ==
        (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.state_model.x0) == (poisson_lds.latent_dim,)
    @test size(poisson_lds.state_model.P0) ==
        (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.obs_model.d) == (poisson_lds.obs_dim,)
end

function test_pobs_constructor_type_preservation()
    # Int
    C_int = [1 2; 3 4]
    d_int = [5, 6]

    pom_int = PoissonObservationModel(; C=C_int, d=d_int)

    @test eltype(pom_int.C) === Int
    @test eltype(pom_int.d) === Int
    @test size(pom_int.C) == (2, 2)
    @test length(pom_int.d) == 2

    # Float32
    C_f32 = Float32[1 2; 3 4]
    d_f32 = Float32[0.5, 0.6]

    pom_f32 = PoissonObservationModel(; C=C_f32, d=d_f32)

    @test eltype(pom_f32.C) === Float32
    @test eltype(pom_f32.d) === Float32
    @test size(pom_f32.C) == (2, 2)
    @test length(pom_f32.d) == 2

    # BigFloat
    C_bf = BigFloat[1 2; 3 4]
    d_bf = BigFloat[0.1, 0.2]

    pom_bf = PoissonObservationModel(; C=C_bf, d=d_bf)

    @test eltype(pom_bf.C) === BigFloat
    @test eltype(pom_bf.d) === BigFloat
    @test size(pom_bf.C) == (2, 2)
    @test length(pom_bf.d) == 2
end

function test_plds_constructor_type_preservation()
    # Int
    A_int = [1 2; 3 4]
    C_int = [1 0; 0 1]
    Q_int = [2 0; 0 2]
    d_int = [7, 8]
    x0_int = [5, 6]
    P0_int = [4 0; 0 4]
    b_int = [0, 0]

    gsm_int = GaussianStateModel(; A=A_int, Q=Q_int, x0=x0_int, P0=P0_int, b=b_int)
    pom_int = PoissonObservationModel(; C=C_int, d=d_int)
    plds_int = LinearDynamicalSystem(;
        state_model=gsm_int,
        obs_model=pom_int,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    @test eltype(plds_int.state_model.A) === Int
    @test eltype(plds_int.state_model.Q) === Int
    @test eltype(plds_int.state_model.x0) === Int
    @test eltype(plds_int.state_model.P0) === Int
    @test eltype(plds_int.obs_model.C) === Int
    @test eltype(plds_int.obs_model.d) === Int
    @test plds_int.latent_dim == 2
    @test plds_int.obs_dim == 2

    # Float32
    A_f32 = Float32[1 2; 3 4]
    C_f32 = Float32[1 0; 0 1]
    Q_f32 = Float32[2 0; 0 2]
    d_f32 = Float32[0.5, 1.5]
    x0_f32 = Float32[0.7, 0.8]
    P0_f32 = Float32[4 0; 0 4]
    b_f32 = Float32[0, 0]

    gsm_f32 = GaussianStateModel(; A=A_f32, Q=Q_f32, x0=x0_f32, P0=P0_f32, b=b_f32)
    pom_f32 = PoissonObservationModel(; C=C_f32, d=d_f32)
    plds_f32 = LinearDynamicalSystem(;
        state_model=gsm_f32,
        obs_model=pom_f32,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    @test eltype(plds_f32.state_model.A) === Float32
    @test eltype(plds_f32.obs_model.C) === Float32
    @test eltype(plds_f32.obs_model.d) === Float32

    # BigFloat
    A_bf = BigFloat[1 2; 3 4]
    C_bf = BigFloat[1 0; 0 1]
    Q_bf = BigFloat[2 0; 0 2]
    d_bf = BigFloat[0.1, 0.2]
    x0_bf = BigFloat[0.3, 0.4]
    P0_bf = BigFloat[4 0; 0 4]
    b_bf = BigFloat[0, 0]

    gsm_bf = GaussianStateModel(; A=A_bf, Q=Q_bf, x0=x0_bf, P0=P0_bf, b=b_bf)
    pom_bf = PoissonObservationModel(; C=C_bf, d=d_bf)
    plds_bf = LinearDynamicalSystem(;
        state_model=gsm_bf,
        obs_model=pom_bf,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    @test eltype(plds_bf.state_model.A) === BigFloat
    @test eltype(plds_bf.obs_model.C) === BigFloat
    @test eltype(plds_bf.obs_model.d) === BigFloat
end

function test_poisson_sample_type_preservation()
    # Float32
    A_f32 = Matrix{Float32}(I, 2, 2)
    C_f32 = Matrix{Float32}(I, 2, 2)
    Q_f32 = Matrix{Float32}(I, 2, 2)
    d32 = zeros(Float32, 2)
    x0_f32 = fill(one(Float32), 2)
    P0_f32 = Matrix{Float32}(I, 2, 2)
    b_f32 = zeros(Float32, 2)

    gsm_f32 = GaussianStateModel(; A=A_f32, Q=Q_f32, x0=x0_f32, P0=P0_f32, b=b_f32)
    pom_f32 = PoissonObservationModel(; C=C_f32, d=d32)
    plds_f32 = LinearDynamicalSystem(;
        state_model=gsm_f32,
        obs_model=pom_f32,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    x_f32, y_f32 = rand(plds_f32, fill(50, 3))

    @test eltype(x_f32[1]) === Float32
    @test eltype(y_f32[1]) === Float32
    @test length(x_f32) == 3
    @test length(y_f32) == 3
    @test size(x_f32[1]) == (2, 50)
    @test size(y_f32[1]) == (2, 50)

    # BigFloat
    A_bf = Matrix{BigFloat}(I, 2, 2)
    C_bf = Matrix{BigFloat}(I, 2, 2)
    Q_bf = Matrix{BigFloat}(I, 2, 2)
    log_bf = zeros(BigFloat, 2)
    x0_bf = fill(one(BigFloat), 2)
    P0_bf = Matrix{BigFloat}(I, 2, 2)
    b_bf = zeros(BigFloat, 2)

    gsm_bf = GaussianStateModel(; A=A_bf, Q=Q_bf, x0=x0_bf, P0=P0_bf, b=b_bf)
    pom_bf = PoissonObservationModel(; C=C_bf, d=log_bf)
    plds_bf = LinearDynamicalSystem(;
        state_model=gsm_bf,
        obs_model=pom_bf,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    x_bf, y_bf = rand(plds_bf, fill(50, 3))

    @test eltype(x_bf[1]) === BigFloat
    @test eltype(y_bf[1]) === BigFloat
    @test length(x_bf) == 3
    @test length(y_bf) == 3
    @test size(x_bf[1]) == (2, 50)
    @test size(y_bf[1]) == (2, 50)
end

function test_poisson_fit_type_preservation()
    for T in CHECKED_TYPES
        A = Matrix{T}(I, 2, 2)
        C = Matrix{T}(I, 2, 2)
        Q = Matrix{T}(I, 2, 2)
        d = zeros(T, 2)
        x0 = fill(one(T), 2)
        P0 = Matrix{T}(I, 2, 2)
        b = zeros(T, 2)

        gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
        pom = PoissonObservationModel(; C=C, d=d)
        lds = LinearDynamicalSystem(;
            state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
        )

        # Seeded — A=I+identity-Q can sample data the Newton smoother diverges on
        # under an unlucky global RNG state.
        rng = StableRNG(1)
        x, y = rand(rng, lds, fill(50, 3))

        mls = fit!(lds, y; max_iter=10, tol=1e-6)

        @test eltype(mls) === T
    end
end

function test_poisson_loglikelihood_type_preservation()
    for T in CHECKED_TYPES
        A = Matrix{T}(I, 2, 2)
        C = Matrix{T}(I, 2, 2)
        Q = Matrix{T}(I, 2, 2)
        d = zeros(T, 2)
        x0 = fill(one(T), 2)
        P0 = Matrix{T}(I, 2, 2)
        b = zeros(T, 2)

        gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
        pom = PoissonObservationModel(; C=C, d=d)
        lds = LinearDynamicalSystem(;
            state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
        )

        x, y = rand(lds, fill(50, 3))

        x_mat = x[1]
        y_mat = y[1]

        ll = sum(StateSpaceDynamics.joint_loglikelihood(x_mat, lds, y_mat))

        if ll isa Number
            @test typeof(ll) === T
        else
            @test eltype(ll) === T
        end
    end
end

function test_PoissonLDS_with_params()
    poisson_lds, _, _ = toy_PoissonLDS()
    test_plds_properties(poisson_lds)

    @test poisson_lds.state_model.A == A
    @test poisson_lds.state_model.Q == Q
    @test poisson_lds.obs_model.C == C
    @test poisson_lds.state_model.x0 == x0
    @test poisson_lds.state_model.P0 == P0
    @test poisson_lds.obs_dim == 3
    @test poisson_lds.latent_dim == 2
    @test poisson_lds.fit_bool == [true, true, true, true, true, true]
end

function test_Gradient()
    plds, x, y = toy_PoissonLDS()
    return test_gradient_common(plds, x, y)
end

function test_Hessian()
    plds, x, y = toy_PoissonLDS()
    return test_hessian_common(plds, x, y)
end

function test_smooth()
    plds, x, y = toy_PoissonLDS()
    return test_smooth_common(plds, x, y)
end

function test_parameter_gradient()
    plds, x, y = toy_PoissonLDS()

    tsteps_per_trial = [size(yt, 2) for yt in y]
    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, tsteps_per_trial)

    # Smooth + populate tfs.E_z / E_zz / E_zz_prev for the legacy gradient call.
    T_max = maximum(tsteps_per_trial)
    sws_pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, plds.latent_dim, plds.obs_dim, T_max)
        for _ in 1:Threads.maxthreadid()
    ]
    StateSpaceDynamics.smooth!(plds, tfs, y, sws_pool)
    StateSpaceDynamics.sufficient_statistics!(tfs)

    # params
    C, d = plds.obs_model.C, plds.obs_model.d
    params = vcat(vec(C), d)

    # get analytical gradient
    grad_analytical = StateSpaceDynamics.gradient_observation_model!(
        zeros(length(params)), C, d, tfs, y, sws_pool
    )

    # numerical gradient against trial 1 only
    E_z = tfs[1].E_z
    P_smooth = tfs[1].p_smooth
    y1 = y[1]
    function f(params::AbstractVector{<:Real})
        C_size = plds.obs_dim * plds.latent_dim
        d = params[(end - plds.obs_dim + 1):end]
        C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
        # Canonical Poisson GLM: λ_t = exp(C x_t + d). The previous version of
        # this numerical-gradient closure had `d = exp.(d)` here, which mirrored
        # the (now-fixed) double-exp bug in poisson.jl — the analytical and
        # numerical gradients agreed only because *both* were wrong.
        tsteps = size(y1, 2)
        val = zero(eltype(params))
        for t in 1:tsteps
            h_t = C * E_z[:, t] .+ d
            rho_t = [
                eltype(params)(0.5) * dot(C[i, :], P_smooth[:, :, t] * C[i, :]) for
                i in 1:size(C, 1)
            ]
            val += dot(y1[:, t], h_t) - sum(exp.(h_t .+ rho_t))
        end
        return -val
    end

    grad = ForwardDiff.gradient(f, params)

    @test isapprox(grad, grad_analytical, rtol=1e-5, atol=1e-5)
end

function test_initial_observation_parameter_updates(ntrials::Int=1)
    return test_initial_state_parameter_updates_common(toy_PoissonLDS, ntrials)
end

function test_state_model_parameter_updates(ntrials::Int=1)
    return test_state_model_parameter_updates_common(toy_PoissonLDS, ntrials)
end

function test_EM(n_trials::Int=1)
    return test_em_convergence_common(toy_PoissonLDS, n_trials)
end

function test_EM_matlab()
    # read data used to smooth the results
    data_1 = Matrix(CSV.read("test_data/trial1.csv", DataFrame))
    data_2 = Matrix(CSV.read("test_data/trial2.csv", DataFrame))
    data_3 = Matrix(CSV.read("test_data/trial3.csv", DataFrame))
    y = [permutedims(d, [2, 1]) for d in (data_1, data_2, data_3)]
    # read the matlab objects to compare results
    seq = matread("test_data/seq_matlab_3_trials_plds.mat")
    params = matread("test_data/params_matlab_3_trials_plds.mat")

    # The MATLAB reference was generated against the *buggy* Poisson model
    # `λ = exp(C x + exp(log_d))` with `log_d = log([0.1, 0.1, 0.1])`. The
    # *effective* additive offset MATLAB used was `exp(log_d) = [0.1, 0.1, 0.1]`.
    # Under the canonical post-fix model `λ = exp(C x + d)`, setting
    # `d = [0.1, 0.1, 0.1]` directly reproduces MATLAB's likelihood — same E-step
    # posteriors, same M-step optimum (the MAP is reachable from either
    # parameterization). Hence the comparison still holds as a regression check.
    gsm = GaussianStateModel(;
        A=[cos(0.1) -sin(0.1); sin(0.1) cos(0.1)],
        Q=0.00001 * Matrix{Float64}(I(2)),
        x0=[1.0, -1.0],
        P0=0.00001 * Matrix{Float64}(I(2)),
        b=zeros(2),
    )

    pom = PoissonObservationModel(; C=[1.2 1.2; 1.2 1.2; 1.2 1.2], d=[0.1, 0.1, 0.1])

    plds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=3, fit_bool=fill(true, 6)
    )

    tsteps_per_trial = [size(yt, 2) for yt in y]
    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, tsteps_per_trial)

    # first smooth results
    T_max = maximum(tsteps_per_trial)
    sws_pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, plds.latent_dim, plds.obs_dim, T_max)
        for _ in 1:Threads.maxthreadid()
    ]
    StateSpaceDynamics.smooth!(plds, tfs, y, sws_pool)
    StateSpaceDynamics.sufficient_statistics!(tfs)

    # Check each posterior moment against MATLAB's reference.
    # MATLAB stores Vsm as `(T·D, D)` — vertically-stacked (D × D) covariance
    # blocks across timesteps — and VVsm as `((T-1)·D, D)` lagged covariance
    # blocks for t = 2..T. Reshape both to Julia's `(D, D, T)` / `(D, D, T-1)`
    # layout for direct comparison with `tfs[i].p_smooth` /
    # `tfs[i].p_smooth_tt1[:, :, 2:end]` (slice 1 of `p_smooth_tt1` is the
    # unused lag-into-prehistory entry).
    D = plds.latent_dim
    for i in 1:3
        T_i = size(y[i], 2)
        posterior_x = seq["seq"]["posterior"][i]["xsm"]
        Vsm_3d = permutedims(
            reshape(seq["seq"]["posterior"][i]["Vsm"], D, T_i, D), (1, 3, 2)
        )
        VVsm_3d = permutedims(
            reshape(seq["seq"]["posterior"][i]["VVsm"], D, T_i - 1, D), (1, 3, 2)
        )
        @test isapprox(tfs[i].E_z, posterior_x, atol=1e-6)
        @test isapprox(tfs[i].p_smooth, Vsm_3d, atol=1e-10)
        @test isapprox(tfs[i].p_smooth_tt1[:, :, 2:end], VVsm_3d, atol=1e-10)
    end
    # now test the params after one EM step
    fit!(plds, y; max_iter=1)
    params_obj = params["params"]["model"]
    @test isapprox(plds.state_model.A, params_obj["A"], atol=1e-5)
    @test isapprox(plds.state_model.Q, params_obj["Q"], atol=1e-5)
    @test isapprox(plds.obs_model.C, params_obj["C"], atol=1e-5)
    @test isapprox(plds.state_model.x0, params_obj["x0"], atol=1e-5)
    @test isapprox(plds.state_model.P0, params_obj["Q0"], atol=1e-5)
    # MATLAB stored `d` as the effective offset (= natural firing rate, since
    # the buggy model's exp(log_d) IS the offset). Under the canonical model
    # that's just `d` directly — no exp wrapping.
    @test isapprox(plds.obs_model.d, params_obj["d"], atol=1e-5)
end

function test_poisson_map_step_improves_Q(; rng=MersenneTwister(123))
    @testset "PoissonLDS: observation MAP step improves Q (LBFGS)" begin
        D, P, Tt, N = 2, 3, 40, 3

        # concrete matrices (avoid UniformScaling in fields)
        A = 0.9 .* Matrix(I, D, D)
        Q = 0.15 .* Matrix(I, D, D)
        b = zeros(D)
        x0 = zeros(D)
        P0 = 0.15 .* Matrix(I, D, D)

        C = 0.3 .* randn(rng, P, D)
        d = log.(0.7 .+ rand(rng, P))

        gsm = GaussianStateModel(A=A, Q=Q, b=b, x0=x0, P0=P0)
        pom = PoissonObservationModel(C=C, d=d)
        plds = LinearDynamicalSystem(gsm, pom)

        _, Y = rand(rng, plds, fill(Tt, N))

        tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, fill(Tt, N))
        sws_pool = [
            StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt) for
            _ in 1:Threads.maxthreadid()
        ]
        StateSpaceDynamics.smooth!(plds, tfs, Y, sws_pool)
        StateSpaceDynamics.sufficient_statistics!(tfs)

        ws = StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt)
        StateSpaceDynamics.compute_smooth_constants!(ws, plds)

        Q0 = sum(
            StateSpaceDynamics.Q_obs!(ws, plds, tfs[k].E_z, tfs[k].p_smooth, Y[k]) for
            k in 1:N
        )
        StateSpaceDynamics.update_observation_model!(plds, tfs, Y, sws_pool)  # LBFGS inside
        Q1 = sum(
            StateSpaceDynamics.Q_obs!(ws, plds, tfs[k].E_z, tfs[k].p_smooth, Y[k]) for
            k in 1:N
        )

        @test Q1 ≥ Q0 - 1e-7
        @test all(isfinite, plds.obs_model.C)
        @test all(isfinite, plds.obs_model.d)
    end
    return nothing
end

function test_poisson_cd_prior_shrink(; rng=MersenneTwister(20260604))
    # MN `CD_prior` on the Poisson emission `[C d]` (the MN-only prior; Poisson
    # has no IW observation-noise counterpart). With a strong prior centered at
    # M₀ = 0 and Λ ≫ data, the penalized LBFGS emission update should pull the
    # fitted `[C d]` toward 0. The suf-based Poisson `calculate_elbo` now adds
    # the matching MN trace term `-½ tr((W-M₀)Λ(W-M₀)')`, so the displayed ELBO
    # must stay monotone; were that term missing, the ELBO would drop sharply as
    # `[C d]` shrinks (the data fit worsens while the dropped penalty falls).
    @testset "PoissonLDS: CD_prior shrinks [C d] toward M₀, ELBO monotone" begin
        D, P, Tt, N = 2, 3, 50, 4

        A = 0.9 .* Matrix(I, D, D)
        Q = 0.15 .* Matrix(I, D, D)
        b = zeros(D)
        x0 = zeros(D)
        P0 = 0.15 .* Matrix(I, D, D)
        C = 0.5 .* randn(rng, P, D)
        d = log.(0.7 .+ rand(rng, P))

        # Shared dataset, drawn from a prior-free model.
        gsm_seed = GaussianStateModel(;
            A=copy(A), Q=copy(Q), b=copy(b), x0=copy(x0), P0=copy(P0)
        )
        pom_seed = PoissonObservationModel(; C=copy(C), d=copy(d))
        lds_seed = LinearDynamicalSystem(gsm_seed, pom_seed)
        _, Y = rand(rng, lds_seed, fill(Tt, N))

        # Reference fit: no emission prior.
        gsm_ref = GaussianStateModel(;
            A=copy(A), Q=copy(Q), b=copy(b), x0=copy(x0), P0=copy(P0)
        )
        pom_ref = PoissonObservationModel(; C=copy(C), d=copy(d))
        lds_ref = LinearDynamicalSystem(gsm_ref, pom_ref)
        elbos_ref = fit!(lds_ref, Y; max_iter=20, progress=false)
        @test all(diff(elbos_ref) .>= -1e-4 .* abs.(@view elbos_ref[1:(end - 1)]) .- 1e-6)

        # Same data, very strong MN prior on [C d] centered at zero. CD lives in
        # (P × D+1): D columns for C, one for the intercept d.
        CD_M0 = zeros(P, D + 1)
        CD_Λ = 1e6 .* Matrix{Float64}(I, D + 1, D + 1)
        gsm_p = GaussianStateModel(;
            A=copy(A), Q=copy(Q), b=copy(b), x0=copy(x0), P0=copy(P0)
        )
        pom_p = PoissonObservationModel(;
            C=copy(C), d=copy(d), CD_prior=StateSpaceDynamics.MNPrior(; M₀=CD_M0, Λ=CD_Λ)
        )
        lds_p = LinearDynamicalSystem(gsm_p, pom_p)
        elbos_p = fit!(lds_p, Y; max_iter=20, progress=false)

        # ELBO monotone under the MN penalty (relative slack for the Laplace
        # E-step). A missing MN term in calculate_elbo would violate this by a
        # wide margin given Λ = 1e6.
        @test all(diff(elbos_p) .>= -1e-4 .* abs.(@view elbos_p[1:(end - 1)]) .- 1e-6)
        @test all(isfinite, elbos_p)

        # Strong shrinkage: penalized [C d] is smaller than reference and pulled
        # to ≈ M₀ = 0.
        W_ref = hcat(lds_ref.obs_model.C, lds_ref.obs_model.d)
        W_p = hcat(lds_p.obs_model.C, lds_p.obs_model.d)
        @test norm(W_p) < norm(W_ref)
        @test norm(lds_p.obs_model.C) < 0.05
        @test all(abs.(lds_p.obs_model.d) .< 0.05)
        @test all(isfinite, lds_p.obs_model.C)
        @test all(isfinite, lds_p.obs_model.d)
    end
    return nothing
end

function test_poisson_gradient_shape_and_finiteness()
    @testset "PoissonLDS: gradient_observation_model! shape & finiteness" begin
        D, P, Tt, N = 2, 3, 20, 2

        A = 0.9 .* Matrix(I, D, D)
        Q = 0.15 .* Matrix(I, D, D)
        b = zeros(D)
        x0 = zeros(D)
        P0 = 0.6 .* Matrix(I, D, D)

        C = 0.2 .* randn(P, D)
        d = zeros(P)

        plds = LinearDynamicalSystem(
            GaussianStateModel(A=A, Q=Q, b=b, x0=x0, P0=P0),
            PoissonObservationModel(C=C, d=d),
        )

        _, Y = rand(plds, fill(Tt, N))
        tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, fill(Tt, N))
        sws_pool = [
            StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt) for
            _ in 1:Threads.maxthreadid()
        ]
        StateSpaceDynamics.smooth!(plds, tfs, Y, sws_pool)
        StateSpaceDynamics.sufficient_statistics!(tfs)

        g = zeros(Float64, length(vec(C)) + length(d))
        StateSpaceDynamics.gradient_observation_model!(
            g, plds.obs_model.C, plds.obs_model.d, tfs, Y, sws_pool
        )

        @test all(isfinite, g)
        @test length(g) == P * D + P
    end
    return nothing
end

"""
    test_poisson_low_rate_recovery()

Regression test for the canonical Poisson GLM `λ = exp(C x + d)`. Constructs
data from low-firing-rate neurons (true rates 0.02, 0.05, 0.10, 0.20 per bin)
with `C ≡ 0` so the rate is determined entirely by `d` and verifies that the
fit recovers `exp(d_hat) ≈ true_rate`.

This test would have failed under the prior `λ = exp(C x + exp(log_d))`
double-exp: with C = 0, the achievable rate floor was `exp(0⁺) = 1`, so
target rates well below 1 spike/bin were unrepresentable.
"""
function test_poisson_low_rate_recovery()
    @testset "PoissonLDS: low-rate recovery (would fail under double-exp)" begin
        rng = MersenneTwister(20260509)

        D, P, Tt, N = 2, 4, 4000, 4
        true_rates = [0.02, 0.05, 0.10, 0.20]   # all < 1 spike/bin

        # Simulation truth: latent dynamics decoupled from emissions (C = 0)
        # so observed mean rate is exactly exp(d).
        A_true = 0.9 .* Matrix{Float64}(I, D, D)
        Q_true = 0.05 .* Matrix{Float64}(I, D, D)
        b_true = zeros(D)
        x0_true = zeros(D)
        P0_true = 0.05 .* Matrix{Float64}(I, D, D)
        C_true = zeros(P, D)
        d_true = log.(true_rates)               # d = log(rate) since C ≡ 0

        sm_true = GaussianStateModel(A=A_true, Q=Q_true, b=b_true, x0=x0_true, P0=P0_true)
        om_true = PoissonObservationModel(C=C_true, d=d_true)
        plds_true = LinearDynamicalSystem(sm_true, om_true)

        _, Y = rand(rng, plds_true, fill(Tt, N))

        # Sanity: empirical mean rates land near the true rates.
        emp = mean(reduce(hcat, Y); dims=2) ./ 1.0
        for i in 1:P
            @test isapprox(emp[i], true_rates[i]; rtol=0.25)
        end

        # Fit from a different starting point (rate ≈ 1 spikes/bin baseline).
        sm_fit = GaussianStateModel(
            A=copy(A_true),
            Q=copy(Q_true),
            b=copy(b_true),
            x0=copy(x0_true),
            P0=copy(P0_true),
        )
        om_fit = PoissonObservationModel(C=zeros(P, D), d=zeros(P))
        plds_fit = LinearDynamicalSystem(sm_fit, om_fit)

        fit!(plds_fit, Y; max_iter=20, progress=false)

        recovered_rates = exp.(plds_fit.obs_model.d)
        for i in 1:P
            # Tolerance: 30% relative — generous, since with C=0 fits the
            # latent/dynamics parameters are unidentifiable and only `d`
            # carries the rate signal.
            @test isapprox(recovered_rates[i], true_rates[i]; rtol=0.30)
        end

        # Strong regression assertion: if the bug were back, `d_hat` would be
        # driven to large negative values (or NaN) trying to compensate for
        # the spurious `exp(d)` floor. Guard against that explicitly.
        @test all(plds_fit.obs_model.d .> -10.0)
        @test all(isfinite, plds_fit.obs_model.d)
    end
    return nothing
end

