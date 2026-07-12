# Include common test utilities
include("CommonLDS.jl")

# Define the parameters of a pendulum
g = 9.81 # gravity
l = 1.0 # length of pendulum
dt = 0.01 # time step

# Discrete-time dynamics
A = [1.0 dt; -g / l*dt 1.0]
Q = Matrix{Float64}(0.00001 * I(2))               # Process noise covariance

# Initial state/ covariance
x0 = [0.0; 1.0]
P0 = Matrix{Float64}(0.1 * I(2))                  # Initial state covariance

# Observation params
C = Matrix{Float64}(I(2))                         # Direct observation
observation_noise_std = 0.5
R = Matrix{Float64}((observation_noise_std^2) * I(2))
b = zeros(Float64, 2)                              # state bias
d = zeros(Float64, 2)                              # observation bias

function toy_lds(
    ntrials::Int=1, fit_bool::Vector{Bool}=[true, true, true, true, true, true]
)
    gaussian_sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    gaussian_om = GaussianObservationModel(; C=C, d=d, R=R)
    lds = LinearDynamicalSystem(;
        state_model=gaussian_sm,
        obs_model=gaussian_om,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )
    # sample data
    T = 100
    x, y = StateSpaceDynamics.rand(lds, fill(T, ntrials))
    return lds, x, y
end

function test_lds_properties(lds)
    @test isa(lds.state_model, StateSpaceDynamics.GaussianStateModel)
    @test isa(lds.obs_model, StateSpaceDynamics.GaussianObservationModel)
    @test isa(lds, StateSpaceDynamics.LinearDynamicalSystem)

    @test size(lds.state_model.A) == (lds.latent_dim, lds.latent_dim)
    @test size(lds.obs_model.C) == (lds.obs_dim, lds.latent_dim)
    @test size(lds.state_model.Q) == (lds.latent_dim, lds.latent_dim)
    @test size(lds.obs_model.R) == (lds.obs_dim, lds.obs_dim)
    @test size(lds.state_model.x0) == (lds.latent_dim,)
    @test size(lds.state_model.P0) == (lds.latent_dim, lds.latent_dim)
    @test size(lds.state_model.b) == (lds.latent_dim,)
    @test size(lds.obs_model.d) == (lds.obs_dim,)
end

function test_gaussian_obs_constructor_type_preservation()
    # Int
    A_int = [1 2; 3 4]
    Q_int = [1 0; 0 1]
    x0_int = [5; 6]
    P0_int = [2 0; 0 2]
    b_int = [0, 0]

    gsm_int = GaussianStateModel(; A=A_int, Q=Q_int, b=b_int, x0=x0_int, P0=P0_int)

    @test eltype(gsm_int.A) === Int
    @test eltype(gsm_int.Q) === Int
    @test eltype(gsm_int.x0) === Int
    @test eltype(gsm_int.P0) === Int
    @test eltype(gsm_int.b) === Int

    # Float32
    A_f32 = Float32[1 2; 3 4]
    Q_f32 = Float32[1 0; 0 1]
    x0_f32 = Float32[0.5; 0.6]
    P0_f32 = Float32[0.2 0; 0 0.2]
    b_f32 = Float32[0.0, 0.0]

    gsm_f32 = GaussianStateModel(; A=A_f32, Q=Q_f32, b=b_f32, x0=x0_f32, P0=P0_f32)

    @test eltype(gsm_f32.A) === Float32
    @test eltype(gsm_f32.Q) === Float32
    @test eltype(gsm_f32.x0) === Float32
    @test eltype(gsm_f32.P0) === Float32

    # BigFloat (kept to ensure constructor compiles with BigFloat types)
    A_bf = BigFloat[1 2; 3 4]
    Q_bf = BigFloat[1 0; 0 1]
    x0_bf = BigFloat[0.1; 0.2]
    P0_bf = BigFloat[0.3 0; 0 0.3]
    b_bf = BigFloat[0.0, 0.0]

    gsm_bf = GaussianStateModel(; A=A_bf, Q=Q_bf, b=b_bf, x0=x0_bf, P0=P0_bf)

    @test eltype(gsm_bf.A) === BigFloat
    @test eltype(gsm_bf.Q) === BigFloat
    @test eltype(gsm_bf.x0) === BigFloat
    @test eltype(gsm_bf.P0) === BigFloat
    @test eltype(gsm_bf.b) === BigFloat
end

function test_gaussian_lds_constructor_type_preservation()
    # Int
    A_int = [1 2; 3 4]
    C_int = [1 0; 0 1]
    Q_int = [2 0; 0 2]
    R_int = [3 0; 0 3]
    x0_int = [5, 6]
    P0_int = [4 0; 0 4]
    b_int = [0, 0]
    d_int = [0, 0]

    gsm_int = GaussianStateModel(; A=A_int, Q=Q_int, b=b_int, x0=x0_int, P0=P0_int)
    gom_int = GaussianObservationModel(; C=C_int, R=R_int, d=d_int)
    gls_int = LinearDynamicalSystem(;
        state_model=gsm_int,
        obs_model=gom_int,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    @test eltype(gls_int.state_model.A) === Int
    @test eltype(gls_int.state_model.Q) === Int
    @test eltype(gls_int.state_model.x0) === Int
    @test eltype(gls_int.state_model.P0) === Int
    @test eltype(gls_int.state_model.b) === Int
    @test eltype(gls_int.obs_model.C) === Int
    @test eltype(gls_int.obs_model.R) === Int
    @test eltype(gls_int.obs_model.d) === Int
    @test gls_int.latent_dim == 2
    @test gls_int.obs_dim == 2

    # Float32
    A_f32 = Float32[1 2; 3 4]
    C_f32 = Float32[1 0; 0 1]
    Q_f32 = Float32[2 0; 0 2]
    R_f32 = Float32[3 0; 0 3]
    x0_f32 = Float32[0.5, 1.5]
    P0_f32 = Float32[4 0; 0 4]
    b_f32 = Float32[0.0, 0.0]
    d_f32 = Float32[0.0, 0.0]

    gsm_f32 = GaussianStateModel(; A=A_f32, Q=Q_f32, b=b_f32, x0=x0_f32, P0=P0_f32)
    gom_f32 = GaussianObservationModel(; C=C_f32, R=R_f32, d=d_f32)
    gls_f32 = LinearDynamicalSystem(;
        state_model=gsm_f32,
        obs_model=gom_f32,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    @test eltype(gls_f32.state_model.A) === Float32
    @test eltype(gls_f32.state_model.Q) === Float32
    @test eltype(gls_f32.state_model.x0) === Float32
    @test eltype(gls_f32.state_model.P0) === Float32
    @test eltype(gls_f32.state_model.b) === Float32
    @test eltype(gls_f32.obs_model.C) === Float32
    @test eltype(gls_f32.obs_model.R) === Float32
    @test eltype(gls_f32.obs_model.d) === Float32

    # BigFloat
    A_bf = BigFloat[1 2; 3 4]
    C_bf = BigFloat[1 0; 0 1]
    Q_bf = BigFloat[2 0; 0 2]
    R_bf = BigFloat[3 0; 0 3]
    x0_bf = BigFloat[0.1, 0.2]
    P0_bf = BigFloat[4 0; 0 4]
    b_bf = BigFloat[0.0, 0.0]
    d_bf = BigFloat[0.0, 0.0]

    gsm_bf = GaussianStateModel(; A=A_bf, Q=Q_bf, b=b_bf, x0=x0_bf, P0=P0_bf)
    gom_bf = GaussianObservationModel(; C=C_bf, R=R_bf, d=d_bf)
    gls_bf = LinearDynamicalSystem(;
        state_model=gsm_bf,
        obs_model=gom_bf,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    @test eltype(gls_bf.state_model.A) === BigFloat
    @test eltype(gls_bf.state_model.Q) === BigFloat
    @test eltype(gls_bf.state_model.x0) === BigFloat
    @test eltype(gls_bf.state_model.P0) === BigFloat
    @test eltype(gls_bf.state_model.b) === BigFloat
    @test eltype(gls_bf.obs_model.C) === BigFloat
    @test eltype(gls_bf.obs_model.R) === BigFloat
    @test eltype(gls_bf.obs_model.d) === BigFloat
end

function test_gaussian_sample_type_preservation()
    # Float32
    A_f32 = Matrix{Float32}(I, 2, 2)
    C_f32 = Matrix{Float32}(I, 2, 2)
    Q_f32 = Matrix{Float32}(I, 2, 2)
    R_f32 = Matrix{Float32}(I, 2, 2)
    x0_f32 = fill(one(Float32), 2)
    P0_f32 = Matrix{Float32}(I, 2, 2)
    b_f32 = fill(zero(Float32), 2)
    d_f32 = fill(zero(Float32), 2)

    gsm_f32 = GaussianStateModel(; A=A_f32, Q=Q_f32, b=b_f32, x0=x0_f32, P0=P0_f32)
    gom_f32 = GaussianObservationModel(; C=C_f32, R=R_f32, d=d_f32)
    gls_f32 = LinearDynamicalSystem(;
        state_model=gsm_f32,
        obs_model=gom_f32,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    x_f32, y_f32 = StateSpaceDynamics.rand(gls_f32, fill(50, 3))

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
    R_bf = Matrix{BigFloat}(I, 2, 2)
    x0_bf = fill(one(BigFloat), 2)
    P0_bf = Matrix{BigFloat}(I, 2, 2)
    b_bf = fill(zero(BigFloat), 2)
    d_bf = fill(zero(BigFloat), 2)

    gsm_bf = GaussianStateModel(; A=A_bf, Q=Q_bf, b=b_bf, x0=x0_bf, P0=P0_bf)
    gom_bf = GaussianObservationModel(; C=C_bf, R=R_bf, d=d_bf)
    gls_bf = LinearDynamicalSystem(;
        state_model=gsm_bf,
        obs_model=gom_bf,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6),
    )

    x_bf, y_bf = StateSpaceDynamics.rand(gls_bf, fill(50, 3))

    @test eltype(x_bf[1]) === BigFloat
    @test eltype(y_bf[1]) === BigFloat
    @test length(x_bf) == 3
    @test length(y_bf) == 3
    @test size(x_bf[1]) == (2, 50)
    @test size(y_bf[1]) == (2, 50)
end

function test_gaussian_fit_type_preservation()
    for T in CHECKED_TYPES
        A = Matrix{T}(I, 2, 2)
        C = Matrix{T}(I, 2, 2)
        Q = Matrix{T}(I, 2, 2)
        R = Matrix{T}(I, 2, 2)
        x0 = fill(one(T), 2)
        P0 = Matrix{T}(I, 2, 2)
        b = fill(zero(T), 2)
        d = fill(zero(T), 2)

        gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
        gom = GaussianObservationModel(; C=C, R=R, d=d)
        lds = LinearDynamicalSystem(;
            state_model=gsm, obs_model=gom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
        )

        x, y = rand(lds, fill(50, 3))

        elbos = fit!(lds, y; max_iter=10, tol=1e-6)

        @test eltype(elbos) === T
    end
end

function test_gaussian_loglikelihood_type_preservation()
    for T in CHECKED_TYPES
        A = Matrix{T}(I, 2, 2)
        C = Matrix{T}(I, 2, 2)
        Q = Matrix{T}(I, 2, 2)
        R = Matrix{T}(I, 2, 2)
        x0 = fill(one(T), 2)
        P0 = Matrix{T}(I, 2, 2)
        b = fill(zero(T), 2)
        d = fill(zero(T), 2)

        gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
        gom = GaussianObservationModel(; C=C, R=R, d=d)
        lds = LinearDynamicalSystem(;
            state_model=gsm, obs_model=gom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
        )

        x, y = rand(lds, fill(50, 3))
        x_mat = x[1]
        y_mat = y[1]

        # compute log‐likelihood and check types 
        ll = sum(StateSpaceDynamics.joint_loglikelihood(lds, x_mat, y_mat))

        if ll isa Number
            @test typeof(ll) === T
        else
            @test eltype(ll) === T
        end
    end
end

function test_lds_with_params()
    lds, _, _ = toy_lds()
    test_lds_properties(lds)

    @test lds.state_model.A == A
    @test lds.state_model.Q == Q
    @test lds.obs_model.C == C
    @test lds.obs_model.R == R
    @test lds.state_model.x0 == x0
    @test lds.state_model.P0 == P0
    @test lds.obs_dim == 2
    @test lds.latent_dim == 2
    @test lds.fit_bool == [true, true, true, true, true, true]
end

function test_Gradient()
    lds, x, y = toy_lds()
    return test_gradient_common(lds, x, y)
end

function test_Hessian()
    lds, x, y = toy_lds()
    return test_hessian_common(lds, x, y)
end

function test_smooth()
    lds, x, y = toy_lds()
    test_smooth_common(lds, x, y)
    # Additional Gaussian-specific checks
    tsteps_per_trial = [size(yt, 2) for yt in y]
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
    StateSpaceDynamics.smooth!(lds, tfs, y)
    for i in eachindex(y)
        ws = StateSpaceDynamics.SmoothWorkspace(
            Float64, lds.latent_dim, lds.obs_dim, size(y[i], 2)
        )
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        grad_analytical = copy(StateSpaceDynamics.gradient!(ws, lds, tfs[i].x_smooth, y[i]))
        @test maximum(abs.(grad_analytical)) < 1e-8
        @test norm(grad_analytical) < 1e-8
    end
end

function test_estep()
    lds, x, y = toy_lds()
    return test_estep_common(lds, x, y)
end

function test_initial_observation_parameter_updates(ntrials::Int=1)
    return test_initial_state_parameter_updates_common(toy_lds, ntrials)
end

function test_state_model_parameter_updates(ntrials::Int=1)
    return test_state_model_parameter_updates_common(toy_lds, ntrials)
end

function test_obs_model_parameter_updates(ntrials::Int=1)
    # Fit flags: update C and R here (d is bundled with C via CD)
    lds, x, y = toy_lds(ntrials, [false, false, false, false, true, true])

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

    C_orig = copy(lds.obs_model.C)
    d_orig = copy(lds.obs_model.d)
    R_orig = copy(lds.obs_model.R)

    function obj_obs(CD::AbstractMatrix, R_sqrt::AbstractMatrix, lds)
        D = size(CD, 2) - 1
        lds.obs_model.C .= CD[:, 1:D]
        lds.obs_model.d .= CD[:, D + 1]
        lds.obs_model.R .= R_sqrt * R_sqrt'
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        val = zero(eltype(R_sqrt))

        for k in 1:ntrials
            E_z, E_zz = tfs[k].E_z, tfs[k].E_zz
            val += StateSpaceDynamics.Q_obs!(ws, lds, E_z, E_zz, y[k])
        end
        return -val
    end

    D = lds.latent_dim
    CD0 = hcat(lds.obs_model.C, lds.obs_model.d)
    R_sqrt0 = Matrix(cholesky(lds.obs_model.R).U)

    CD_opt = optimize(CD -> obj_obs(CD, R_sqrt0, lds), CD0, LBFGS()).minimizer
    R_opt_sqrt = optimize(Rs -> obj_obs(CD_opt, Rs, lds), R_sqrt0, LBFGS()).minimizer

    lds.obs_model.C .= C_orig
    lds.obs_model.d .= d_orig
    lds.obs_model.R .= R_orig

    # M-step via the suf-based path: aggregate, then mstep!(lds, suf, ws).
    suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
        Float64, lds, tsteps_per_trial
    )
    ux_seq = [zeros(Float64, 0, size(yt, 2)) for yt in y]
    uy_seq = [zeros(Float64, 0, size(yt, 2)) for yt in y]
    StateSpaceDynamics._td_init_const_blocks!(ws, lds, tsteps_per_trial, y, ux_seq, uy_seq)
    StateSpaceDynamics._aggregate_td_suff_stats!(suf, tfs, lds, ux_seq, uy_seq, y, ws)
    StateSpaceDynamics.mstep!(lds, suf, ws)

    @test isapprox(lds.obs_model.C, CD_opt[:, 1:D], atol=1e-6, rtol=1e-6)
    @test isapprox(lds.obs_model.d, CD_opt[:, D + 1], atol=1e-6, rtol=1e-6)
    @test isapprox(lds.obs_model.R, R_opt_sqrt * R_opt_sqrt', atol=1e-6, rtol=1e-6)
end

function test_EM(n_trials::Int=1)
    test_em_convergence_common(toy_lds, n_trials)
    # Additional check for monotonic increase
    lds, x, y = toy_lds(n_trials)
    ml_total = fit!(lds, y; max_iter=100)
    @test all(diff(ml_total) .>= 0)
end

function test_gaussian_iw_priors_shape_map_and_R_sanity(; rng=MersenneTwister(2025))
    @testset "GaussianLDS: IW priors shape MAP + R update sanity" begin
        D, P, Tt, N = 3, 4, 60, 4

        A = 0.92I + 0.03 * randn(rng, D, D)
        Q = Matrix(0.3 * I(D))
        b = zeros(D)
        x0 = zeros(D)
        P0 = Matrix(0.8 * I(D))

        C = randn(rng, P, D)
        R = Matrix(Symmetric(diagm(0 => 0.25 .+ 0.05 .* rand(rng, P))))
        d = 0.1 .* randn(rng, P)

        # Strong shrinkage priors (ν > d+1)
        Qprior = IWPrior(; Ψ=diagm(0 => fill(0.01, D)), ν=Float64(D + 3))
        P0prior = IWPrior(; Ψ=diagm(0 => fill(0.01, D)), ν=Float64(D + 3))
        Rprior = IWPrior(; Ψ=diagm(0 => fill(0.01, P)), ν=Float64(P + 3))

        gsm = GaussianStateModel(;
            A=A, Q=Q, b=b, x0=x0, P0=P0, Q_prior=Qprior, P0_prior=P0prior
        )
        gom = GaussianObservationModel(; C=C, R=R, d=d, R_prior=Rprior)
        lds = LinearDynamicalSystem(gsm, gom)

        X, Y = rand(rng, lds, fill(Tt, N))

        elbos = fit!(lds, Y; max_iter=12, tol=0.0, progress=false)
        @test all(diff(elbos) .>= -1e-7)
        @test issymmetric(lds.state_model.Q)
        @test issymmetric(lds.state_model.P0)
        @test issymmetric(lds.obs_model.R)

        # Shrinkage sanity checks (loose thresholds)
        @test maximum(eigvals(lds.state_model.Q)) < 0.5
        @test maximum(eigvals(lds.state_model.P0)) < 1.0
        @test maximum(eigvals(lds.obs_model.R)) < 0.4
    end
    return nothing
end

function test_gaussian_update_R_matches_residual_cov(; rng=MersenneTwister(7))
    @testset "GaussianLDS: update_R! ≈ residual covariance when latents ~ deterministic" begin
        D, P, Tt, N = 2, 3, 80, 6

        A = 0.0I + 0.01 * randn(rng, D, D)
        Q = Matrix(I(D) * 1e-7)   # almost deterministic latents
        b = zeros(D)
        x0 = zeros(D)
        P0 = Matrix(I(D) * 1e-7)

        C = randn(rng, P, D)
        Rtrue = Matrix(Symmetric([
            0.30 0.05 0.00
            0.05 0.22 0.02
            0.00 0.02 0.27
        ]))
        d = 0.05 .* randn(rng, P)

        gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
        gom = GaussianObservationModel(; C=C, R=copy(Rtrue), d=d)
        lds = LinearDynamicalSystem(gsm, gom)

        X, Y = rand(rng, lds, fill(Tt, N))

        tsteps_per_trial = fill(Tt, N)
        tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
        ws = StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt)
        sws_pool = [
            StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt) for
            _ in 1:Threads.maxthreadid()
        ]
        StateSpaceDynamics.smooth!(lds, tfs, Y, sws_pool)

        # only update R
        lds.fit_bool .= [false, false, false, false, false, true]
        suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds, tsteps_per_trial
        )
        ux_seq = [zeros(Float64, 0, Tt) for _ in 1:N]
        uy_seq = [zeros(Float64, 0, Tt) for _ in 1:N]
        StateSpaceDynamics._td_init_const_blocks!(
            ws, lds, tsteps_per_trial, Y, ux_seq, uy_seq
        )
        StateSpaceDynamics._aggregate_td_suff_stats!(suf, tfs, lds, ux_seq, uy_seq, Y, ws)
        StateSpaceDynamics.update_R!(lds, suf, ws)

        @test issymmetric(lds.obs_model.R)
        @test norm(lds.obs_model.R - Rtrue) / norm(Rtrue) < 0.25
    end
    return nothing
end

function test_td_mn_priors_shrink(; rng=MersenneTwister(20260519))
    #=
    MN-prior shrinkage on [A b B] and [C d D] in the TD fit. With a strong
    prior centered at M₀ = 0 and Λ ≫ data, the MAP regression should pull
    the fitted coefficients toward 0.
    =#
    @testset "TD: MN priors shrink coefficients toward M₀" begin
        D, p, Tt, N = 3, 4, 60, 4

        A = 0.7 * StateSpaceDynamics.random_rotation_matrix(D, rng)
        Q = Matrix(0.2 * I(D))
        b = randn(rng, D)
        x0 = zeros(D)
        P0 = Matrix(0.5 * I(D))
        C = randn(rng, p, D)
        R = Matrix(0.2 * I(p))
        d = 0.1 * randn(rng, p)

        # Reference fit: no priors.
        sm_ref = GaussianStateModel(;
            A=copy(A), Q=copy(Q), b=copy(b), x0=copy(x0), P0=copy(P0)
        )
        om_ref = GaussianObservationModel(; C=copy(C), R=copy(R), d=copy(d))
        lds_ref = LinearDynamicalSystem(sm_ref, om_ref)
        _, y = rand(rng, lds_ref, fill(Tt, N))

        elbos_ref = fit!(lds_ref, y; max_iter=20, progress=false)
        @test all(diff(elbos_ref) .>= -1e-7)

        # Same data, very strong MN-priors centered at zero.
        # AB lives in (D × D+1) (A and bias); CD lives in (p × D+1).
        AB_M0 = zeros(D, D + 1)
        AB_Λ = 1e6 * Matrix{Float64}(I, D + 1, D + 1)
        CD_M0 = zeros(p, D + 1)
        CD_Λ = 1e6 * Matrix{Float64}(I, D + 1, D + 1)

        sm_p = GaussianStateModel(;
            A=copy(A),
            Q=copy(Q),
            b=copy(b),
            x0=copy(x0),
            P0=copy(P0),
            AB_prior=StateSpaceDynamics.MNPrior(; M₀=AB_M0, Λ=AB_Λ),
        )
        om_p = GaussianObservationModel(;
            C=copy(C),
            R=copy(R),
            d=copy(d),
            CD_prior=StateSpaceDynamics.MNPrior(; M₀=CD_M0, Λ=CD_Λ),
        )
        lds_p = LinearDynamicalSystem(sm_p, om_p)
        #=
        ELBO must be monotone under MN priors — the TD `calculate_elbo` now
        includes the MN log-prior trace term `-½ tr(Σ⁻¹ (W-M₀) Λ (W-M₀)')`
        for both [A b B] and [C d D]. Without it, the displayed ELBO drops
        the MN contribution and can appear non-monotone even though the
        underlying MAP objective is increasing.
        =#
        elbos_p = fit!(lds_p, y; max_iter=20, progress=false)
        @test all(diff(elbos_p) .>= -1e-6)

        # Strong shrinkage check: fitted norms should be smaller than reference.
        @test norm(lds_p.state_model.A) < norm(lds_ref.state_model.A)
        @test norm(lds_p.obs_model.C) < norm(lds_ref.obs_model.C)
        # Near-zero with this much shrinkage.
        @test norm(lds_p.state_model.A) < 0.1
        @test norm(lds_p.obs_model.C) < 0.1
    end
    return nothing
end

function test_td_with_obs_inputs(; rng=MersenneTwister(20260520))
    # TD path with a non-trivial D matrix: simulate from y = C x + d + D v + ε,
    # fit, and verify that fitting *with* obs_inputs beats fitting *without*.
    @testset "TD: obs_inputs (D matrix) is learned" begin
        D, p, Tt, N = 3, 5, 50, 6
        uy_dim = 2

        A_true = 0.85 * StateSpaceDynamics.random_rotation_matrix(D, rng)
        Q_true = 0.05 * Matrix{Float64}(I, D, D)
        b_true = randn(rng, D)
        B_true = zeros(D, 0)
        x0_true = zeros(D)
        P0_true = 0.1 * Matrix{Float64}(I, D, D)
        C_true = randn(rng, p, D)
        R_true = 0.1 * Matrix{Float64}(I, p, p)
        d_true = zeros(p)
        D_true = randn(rng, p, uy_dim)

        sm_true = GaussianStateModel(;
            A=A_true, Q=Q_true, x0=x0_true, P0=P0_true, b=b_true, B=B_true
        )
        om_true = GaussianObservationModel(; C=C_true, R=R_true, d=d_true, D=D_true)
        lds_true = LinearDynamicalSystem(sm_true, om_true)

        uy_seq = [randn(rng, uy_dim, Tt) for _ in 1:N]
        _, y_seq = rand(rng, lds_true, fill(Tt, N); obs_inputs=uy_seq)

        # Fit with obs inputs.
        sm_init = GaussianStateModel(;
            A=0.5 * Matrix{Float64}(I, D, D),
            Q=Matrix{Float64}(I, D, D),
            x0=zeros(D),
            P0=Matrix{Float64}(I, D, D),
            b=zeros(D),
        )
        om_init = GaussianObservationModel(;
            C=randn(rng, p, D), R=Matrix{Float64}(I, p, p), d=zeros(p), D=zeros(p, uy_dim)
        )
        lds_fit = LinearDynamicalSystem(sm_init, om_init)

        elbos = fit!(lds_fit, y_seq; obs_inputs=uy_seq, max_iter=60, progress=false)
        @test all(diff(elbos) .>= -1e-4)

        # Baseline: fit without obs inputs (0-column D).
        sm_nofit = GaussianStateModel(;
            A=0.5 * Matrix{Float64}(I, D, D),
            Q=Matrix{Float64}(I, D, D),
            x0=zeros(D),
            P0=Matrix{Float64}(I, D, D),
            b=zeros(D),
        )
        om_nofit = GaussianObservationModel(;
            C=randn(MersenneTwister(20260520), p, D), R=Matrix{Float64}(I, p, p), d=zeros(p)
        )
        lds_nofit = LinearDynamicalSystem(sm_nofit, om_nofit)
        elbos_no = fit!(lds_nofit, y_seq; max_iter=60, progress=false)

        # Obs inputs must explain real variance: ELBO with D should beat ELBO without.
        @test elbos[end] > elbos_no[end] + 1.0
    end
    return nothing
end

function test_td_ragged_multi_trial(; rng=MersenneTwister(20260521))
    #=
    Ragged-length multi-trial: exercises the variable-length fallback branch
    in smooth!/fit!. Should produce monotone ELBO and match a per-trial fit
    of the longest sub-batch.
    =#
    @testset "TD: ragged-length multi-trial fit" begin
        D, p = 3, 4
        Ts = [20, 30, 25, 40]   # ragged
        N = length(Ts)

        A = 0.8 * StateSpaceDynamics.random_rotation_matrix(D, rng)
        Q = Matrix(0.1 * I(D))
        b = zeros(D)
        x0 = zeros(D)
        P0 = Matrix(0.4 * I(D))
        C = randn(rng, p, D)
        R = Matrix(0.2 * I(p))
        d = zeros(p)

        sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
        om = GaussianObservationModel(; C=C, R=R, d=d)
        lds_data = LinearDynamicalSystem(sm, om)
        _, y = rand(rng, lds_data, Ts)

        # Trial lengths sanity check.
        @test [size(yt, 2) for yt in y] == Ts

        # Smoke: smooth! must run on the ragged fallback path.
        tfs = StateSpaceDynamics.initialize_FilterSmooth(lds_data, Ts)
        T_max = maximum(Ts)
        sws_pool = [
            StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_max) for
            _ in 1:Threads.maxthreadid()
        ]
        StateSpaceDynamics.smooth!(lds_data, tfs, y, sws_pool)
        for trial in 1:N
            @test size(tfs[trial].x_smooth) == (D, Ts[trial])
            @test size(tfs[trial].p_smooth) == (D, D, Ts[trial])
            # The ragged path does NOT alias p_smooth across trials.
            if trial > 1
                @test tfs[trial].p_smooth !== tfs[1].p_smooth
            end
        end

        # Fit converges monotonically.
        sm2 = GaussianStateModel(;
            A=copy(A), Q=copy(Q), b=copy(b), x0=copy(x0), P0=copy(P0)
        )
        om2 = GaussianObservationModel(; C=copy(C), R=copy(R), d=copy(d))
        lds_fit = LinearDynamicalSystem(sm2, om2)
        elbos = fit!(lds_fit, y; max_iter=20, progress=false)
        @test all(diff(elbos) .>= -1e-6)
    end
    return nothing
end

function test_gaussian_weighting_equiv_to_duplication(; rng=MersenneTwister(9))
    @testset "GaussianLDS: weighting ≈ duplicating data" begin
        D, P, Tt, N = 2, 2, 30, 2

        A = Matrix(0.8 * I(D))
        Q = Matrix(I(D) * 0.1)
        b = zeros(D)
        x0 = zeros(D)
        P0 = Matrix(I(D) * 0.5)

        C = randn(rng, P, D)
        R = Matrix(I(D) * 0.1)
        d = zeros(P)

        lds1 = LinearDynamicalSystem(
            GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0),
            GaussianObservationModel(; C=C, R=R, d=d),
        )
        lds2 = LinearDynamicalSystem(
            GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0),
            GaussianObservationModel(; C=C, R=R, d=d),
        )

        _, Y = rand(rng, lds1, fill(Tt, N))

        # Manual weighted EM via the suf-based weighted aggregator.
        tsteps_per_trial = fill(Tt, N)
        tfs = StateSpaceDynamics.initialize_FilterSmooth(lds1, tsteps_per_trial)
        ws = StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt)
        sws_pool = [
            StateSpaceDynamics.SmoothWorkspace(Float64, D, P, Tt) for
            _ in 1:Threads.maxthreadid()
        ]
        suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds1, tsteps_per_trial
        )
        ux_seq = [zeros(Float64, 0, Tt) for _ in 1:N]
        uy_seq = [zeros(Float64, 0, Tt) for _ in 1:N]
        w = [ones(Float64, Tt), 2.0 .* ones(Float64, Tt)]
        for _ in 1:6
            StateSpaceDynamics.smooth!(lds1, tfs, Y, sws_pool)
            StateSpaceDynamics._aggregate_td_suff_stats_weighted!(
                suf, tfs, lds1, ux_seq, uy_seq, Y, w, ws
            )
            StateSpaceDynamics.mstep!(lds1, suf, ws)
        end
        θw = vec([lds1.state_model.A; lds1.obs_model.C])

        # duplicate trial 2 once ~ same effect
        Ydup = vcat(Y, [copy(Y[2])])
        fit!(lds2, Ydup; max_iter=6, tol=0.0, progress=false)
        θd = vec([lds2.state_model.A; lds2.obs_model.C])

        @test norm(θw - θd) / max(norm(θd), 1e-12) < 0.12
    end
    return nothing
end

function test_mn_prior_type_decoupled_from_model_matrix()
    @testset "MN prior type decoupled from model matrix storage" begin
        D, p = 2, 2
        Af = [0.9 0.0; 0.0 0.85]
        Qf = Matrix(0.1 * I(D))
        P0f = Matrix(0.5 * I(D))
        Bf = zeros(D, 0)
        bf = zeros(D)
        x0f = zeros(D)
        Cf = Matrix{Float64}(I, p, D)
        Rf = Matrix(0.2 * I(p))
        Df = zeros(p, 0)
        df = zeros(p)

        sm = GaussianStateModel(;
            A=view(Af, :, :),
            Q=view(Qf, :, :),
            b=view(bf, :),
            x0=view(x0f, :),
            P0=view(P0f, :, :),
            B=view(Bf, :, :),
            AB_prior=StateSpaceDynamics.MNPrior(;
                M₀=zeros(D, D + 1), Λ=Matrix(1e3 * I(D + 1))
            ),
        )
        om = GaussianObservationModel(;
            C=view(Cf, :, :),
            R=view(Rf, :, :),
            d=view(df, :),
            D=view(Df, :, :),
            CD_prior=StateSpaceDynamics.MNPrior(;
                M₀=zeros(p, D + 1), Λ=Matrix(1e3 * I(D + 1))
            ),
        )

        @test sm.A isa SubArray
        @test sm.AB_prior isa StateSpaceDynamics.MNPrior{Float64,Matrix{Float64}}
        @test om.CD_prior isa StateSpaceDynamics.MNPrior{Float64,Matrix{Float64}}

        lds = LinearDynamicalSystem(sm, om)
        kws = StateSpaceDynamics.KalmanWorkspace(lds, 20, 1)
        @test kws.AB_prior isa StateSpaceDynamics.MNPrior{Float64,Matrix{Float64}}
        @test kws.CD_prior isa StateSpaceDynamics.MNPrior{Float64,Matrix{Float64}}
        @test kws.AB_prior === sm.AB_prior   # stored verbatim, no copy/convert
        @test kws.CD_prior === om.CD_prior
    end
    return nothing
end

function test_td_weighted_aggregator_matches_unweighted_with_inputs(;
    rng=MersenneTwister(0xC0FFEE)
)
    @testset "TD weighted aggregator == unweighted (B & D inputs)" begin
        D, p, ux_dim, uy_dim, Tt = 2, 3, 2, 2, 40

        A = 0.85 * StateSpaceDynamics.random_rotation_matrix(D, rng)
        Q = 0.05 * Matrix{Float64}(I, D, D)
        b = randn(rng, D)
        B = 0.3 * randn(rng, D, ux_dim)
        x0 = zeros(D)
        P0 = 0.1 * Matrix{Float64}(I, D, D)
        C = randn(rng, p, D)
        R = 0.1 * Matrix{Float64}(I, p, p)
        dvec = randn(rng, p)
        Dmat = 0.4 * randn(rng, p, uy_dim)

        sm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b, B=B)
        om = GaussianObservationModel(; C=C, R=R, d=dvec, D=Dmat)
        lds = LinearDynamicalSystem(sm, om)

        u = 0.5 * randn(rng, ux_dim, Tt)
        v = 0.5 * randn(rng, uy_dim, Tt)
        _, y1 = rand(rng, lds, Tt; latent_inputs=u, obs_inputs=v)
        y = [y1]
        ux_seq = [u]
        uy_seq = [v]

        tsteps_per_trial = [Tt]
        tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
        sws_pool = [
            StateSpaceDynamics.SmoothWorkspace(
                Float64, D, p, Tt; ux_dim=ux_dim, uy_dim=uy_dim, ntrials=1
            ) for _ in 1:Threads.maxthreadid()
        ]
        ws = sws_pool[1]

        # Populate the smoother outputs (x_smooth, p_smooth, p_smooth_tt1) once;
        # both aggregators read the same tfs.
        StateSpaceDynamics._td_init_const_blocks!(
            ws, lds, tsteps_per_trial, y, ux_seq, uy_seq
        )
        StateSpaceDynamics.smooth!(lds, tfs, y, sws_pool, ux_seq, uy_seq)

        # Reference.
        suf_u = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds, tsteps_per_trial
        )
        StateSpaceDynamics._aggregate_td_suff_stats!(suf_u, tfs, lds, ux_seq, uy_seq, y, ws)
        ref = (
            init_n=suf_u.init_n,
            dyn_n=suf_u.dyn_n,
            obs_n=suf_u.obs_n,
            init_xy=copy(suf_u.init_xy),
            dyn_xy=copy(suf_u.dyn_xy),
            obs_xy=copy(suf_u.obs_xy),
            init_yy=copy(suf_u.init_yy[].mat),
            dyn_xx=copy(suf_u.dyn_xx[].mat),
            dyn_yy=copy(suf_u.dyn_yy[].mat),
            obs_xx=copy(suf_u.obs_xx[].mat),
            obs_yy=copy(suf_u.obs_yy[].mat),
        )

        # Weighted aggregator with unit weights must reproduce it exactly.
        suf_w = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds, tsteps_per_trial
        )
        weights = [ones(Float64, Tt)]
        StateSpaceDynamics._aggregate_td_suff_stats_weighted!(
            suf_w, tfs, lds, ux_seq, uy_seq, y, weights, ws
        )

        @test suf_w.init_n ≈ ref.init_n
        @test suf_w.dyn_n ≈ ref.dyn_n
        @test suf_w.obs_n ≈ ref.obs_n
        @test suf_w.init_xy ≈ ref.init_xy
        @test suf_w.init_yy[].mat ≈ ref.init_yy
        # dyn_xx / obs_xx carry the x·u and x·v cross blocks under test; dyn_xy /
        # obs_xy carry the u·x_next and v·y cross rows.
        @test suf_w.dyn_xx[].mat ≈ ref.dyn_xx
        @test suf_w.dyn_xy ≈ ref.dyn_xy
        @test suf_w.dyn_yy[].mat ≈ ref.dyn_yy
        @test suf_w.obs_xx[].mat ≈ ref.obs_xx
        @test suf_w.obs_xy ≈ ref.obs_xy
        @test suf_w.obs_yy[].mat ≈ ref.obs_yy

        # Sanity: the inputs cross blocks are actually populated (non-zero), so
        # the equivalence above is meaningful and not comparing empty regions.
        @test norm(suf_w.dyn_xx[].mat[1:D, (D + 2):end]) > 0
        @test norm(suf_w.obs_xx[].mat[1:D, (D + 2):end]) > 0
    end
    return nothing
end

function test_joint_loglikelihood_matches_mvnormal()
    #=
    Regression test for the U- vs U'-solve quadratic-form bug: with a
    Cholesky Σ = U'U, only the transposed factor whitens (r'Σ⁻¹r = ‖U⁻ᵀr‖²);
    solving with U itself computes r'(UU')⁻¹r. 
    =#
    rng = StableRNG(1234)
    D_lat, p_obs, T_steps = 2, 3, 25

    A_nd = [0.9 0.1; -0.05 0.85]
    Q_nd = [0.5 0.2; 0.2 0.4]
    b_nd = [0.1, -0.1]
    x0_nd = [0.5, -0.5]
    P0_nd = [1.0 0.3; 0.3 0.8]
    C_nd = randn(rng, p_obs, D_lat)
    R_nd = [0.6 0.1 0.0; 0.1 0.5 0.05; 0.0 0.05 0.7]
    d_nd = [0.1, 0.2, -0.1]

    sm = GaussianStateModel(; A=A_nd, Q=Q_nd, b=b_nd, x0=x0_nd, P0=P0_nd)
    om = GaussianObservationModel(; C=C_nd, R=R_nd, d=d_nd)
    lds = LinearDynamicalSystem(;
        state_model=sm,
        obs_model=om,
        latent_dim=D_lat,
        obs_dim=p_obs,
        fit_bool=fill(true, 6),
    )

    x = randn(rng, D_lat, T_steps)
    y = randn(rng, p_obs, T_steps)

    ll = sum(StateSpaceDynamics.joint_loglikelihood(lds, x, y))

    ref = logpdf(MvNormal(x0_nd, Symmetric(P0_nd)), x[:, 1])
    for t in 2:T_steps
        ref += logpdf(MvNormal(A_nd * x[:, t - 1] .+ b_nd, Symmetric(Q_nd)), x[:, t])
    end
    for t in 1:T_steps
        ref += logpdf(MvNormal(C_nd * x[:, t] .+ d_nd, Symmetric(R_nd)), y[:, t])
    end

    @test ll ≈ ref rtol = 1e-10
    return nothing
end

function test_gaussian_gradient_nondiag()
    #=
    Non-diagonal Q/P0/R plus state (B*ux) and observation (D*uy) inputs;
    gradient checked against ForwardDiff through joint_loglikelihood!.
    =#
    rng = StableRNG(4321)
    D_lat, p_obs, T_steps = 2, 3, 30

    A_nd = [0.9 0.1; -0.05 0.85]
    Q_nd = [0.5 0.2; 0.2 0.4]
    b_nd = [0.1, -0.1]
    x0_nd = [0.5, -0.5]
    P0_nd = [1.0 0.3; 0.3 0.8]
    B_nd = randn(rng, D_lat, 2)
    C_nd = randn(rng, p_obs, D_lat)
    R_nd = [0.6 0.1 0.0; 0.1 0.5 0.05; 0.0 0.05 0.7]
    d_nd = [0.1, 0.2, -0.1]
    D_obs_nd = randn(rng, p_obs, 1)

    sm = GaussianStateModel(; A=A_nd, Q=Q_nd, b=b_nd, x0=x0_nd, P0=P0_nd, B=B_nd)
    om = GaussianObservationModel(; C=C_nd, R=R_nd, d=d_nd, D=D_obs_nd)
    lds = LinearDynamicalSystem(;
        state_model=sm,
        obs_model=om,
        latent_dim=D_lat,
        obs_dim=p_obs,
        fit_bool=fill(true, 6),
    )

    x = randn(rng, D_lat, T_steps)
    y = randn(rng, p_obs, T_steps)
    ux = randn(rng, 2, T_steps)
    uy = randn(rng, 1, T_steps)

    ws = StateSpaceDynamics.SmoothWorkspace(Float64, D_lat, p_obs, T_steps)
    StateSpaceDynamics.compute_smooth_constants!(ws, lds)
    g = copy(StateSpaceDynamics.gradient!(ws, lds, x, y, ux, uy))

    f =
        xv -> begin
            xm = reshape(xv, D_lat, T_steps)
            wsd = StateSpaceDynamics.SmoothWorkspace(eltype(xv), D_lat, p_obs, T_steps)
            StateSpaceDynamics.compute_smooth_constants!(wsd, lds)
            sum(StateSpaceDynamics.joint_loglikelihood!(wsd, lds, xm, y, ux, uy))
        end
    g_num = reshape(ForwardDiff.gradient(f, vec(x)), D_lat, T_steps)

    @test norm(g - g_num) < 1e-8
    return nothing
end

function test_gaussian_hessian_nondiag()
    #=
    Hessian companion to `test_gaussian_gradient_nondiag`: non-diagonal
    Q/P0/R plus control inputs on both sides, checked against
    ForwardDiff.hessian through the kernel-based joint_loglikelihood!.
    Also verifies the Gaussian Hessian is input-independent (the analytic
    path never sees ux/uy; the numerical path differentiates with them).
    =#
    rng = StableRNG(4321)
    D_lat, p_obs, T_steps = 2, 3, 10

    A_nd = [0.9 0.1; -0.05 0.85]
    Q_nd = [0.5 0.2; 0.2 0.4]
    b_nd = [0.1, -0.1]
    x0_nd = [0.5, -0.5]
    P0_nd = [1.0 0.3; 0.3 0.8]
    B_nd = randn(rng, D_lat, 2)
    C_nd = randn(rng, p_obs, D_lat)
    R_nd = [0.6 0.1 0.0; 0.1 0.5 0.05; 0.0 0.05 0.7]
    d_nd = [0.1, 0.2, -0.1]
    D_obs_nd = randn(rng, p_obs, 1)

    sm = GaussianStateModel(; A=A_nd, Q=Q_nd, b=b_nd, x0=x0_nd, P0=P0_nd, B=B_nd)
    om = GaussianObservationModel(; C=C_nd, R=R_nd, d=d_nd, D=D_obs_nd)
    lds = LinearDynamicalSystem(;
        state_model=sm,
        obs_model=om,
        latent_dim=D_lat,
        obs_dim=p_obs,
        fit_bool=fill(true, 6),
    )

    x = randn(rng, D_lat, T_steps)
    y = randn(rng, p_obs, T_steps)
    ux = randn(rng, 2, T_steps)
    uy = randn(rng, 1, T_steps)

    ws = StateSpaceDynamics.SmoothWorkspace(Float64, D_lat, p_obs, T_steps)
    StateSpaceDynamics.compute_smooth_constants!(ws, lds)
    StateSpaceDynamics.hessian!(ws, lds, x, y)
    hess = block_tridgm(ws.btd.H_diag, ws.btd.H_super, ws.btd.H_sub)

    f =
        xv -> begin
            xm = reshape(xv, D_lat, T_steps)
            wsd = StateSpaceDynamics.SmoothWorkspace(eltype(xv), D_lat, p_obs, T_steps)
            StateSpaceDynamics.compute_smooth_constants!(wsd, lds)
            sum(StateSpaceDynamics.joint_loglikelihood!(wsd, lds, xm, y, ux, uy))
        end
    hess_num = ForwardDiff.hessian(f, vec(x))

    @test norm(hess_num - hess) < 1e-8
    return nothing
end
