# define parameters for a PoissonLDS
x0 = [1.0, -1.0]
P0 = Matrix(Diagonal([0.1, 0.1]))  # Fixed: was p0, now P0
A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
Q = Matrix(Diagonal([0.1, 0.1]))
C = [0.6 0.6; 0.6 0.6; 0.6 0.6] .* 2
log_d = log.([0.1, 0.1, 0.1])

function toy_PoissonLDS(
    ntrials::Int=1, fit_bool::Vector{Bool}=[true, true, true, true, true, true]
)
    gaussian_sm = GaussianStateModel(A=A, Q=Q, x0=x0, P0=P0)
    poisson_om = PoissonObservationModel(C=C, log_d=log_d)  # Fixed: was poisson_sm
    poisson_lds = LinearDynamicalSystem(
        state_model=gaussian_sm, 
        obs_model=poisson_om,  
        latent_dim=2, 
        obs_dim=3, 
        fit_bool=fill(true, 6)
    )

    # sample data
    T = 100
    x, y = rand(poisson_lds; tsteps=T, ntrials=ntrials) # 100 timepoints, ntrials

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
    @test size(poisson_lds.obs_model.log_d) == (poisson_lds.obs_dim,)
end

function test_pobs_constructor_type_preservation()
    # Int
    C_int = [1 2; 3 4]
    log_d_int = [5, 6]

    pom_int = PoissonObservationModel(
        C = C_int,
        log_d = log_d_int
    )

    @test eltype(pom_int.C) === Int
    @test eltype(pom_int.log_d) === Int
    @test size(pom_int.C) == (2, 2)
    @test length(pom_int.log_d) == 2

    # Float32
    C_f32     = Float32[1 2; 3 4]
    log_d_f32 = Float32[0.5, 0.6]

    pom_f32 = PoissonObservationModel(
        C = C_f32,
        log_d = log_d_f32
    )

    @test eltype(pom_f32.C) === Float32
    @test eltype(pom_f32.log_d) === Float32
    @test size(pom_f32.C) == (2, 2)
    @test length(pom_f32.log_d) == 2

    # BigFloat
    C_bf = BigFloat[1 2; 3 4]
    log_d_bf = BigFloat[0.1, 0.2]

    pom_bf = PoissonObservationModel(
        C = C_bf,
        log_d = log_d_bf
    )

    @test eltype(pom_bf.C) === BigFloat
    @test eltype(pom_bf.log_d) === BigFloat
    @test size(pom_bf.C) == (2, 2)
    @test length(pom_bf.log_d) == 2
end

function test_plds_constructor_type_preservation()
    # Int
    A_int = [1 2; 3 4]
    C_int = [1 0; 0 1]
    Q_int = [2 0; 0 2]
    log_d_int = [7, 8]
    x0_int = [5, 6]
    P0_int = [4 0; 0 4]

    gsm_int = GaussianStateModel(A=A_int, Q=Q_int, x0=x0_int, P0=P0_int)
    pom_int = PoissonObservationModel(C=C_int, log_d=log_d_int)
    plds_int = LinearDynamicalSystem(
        state_model=gsm_int,
        obs_model=pom_int,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6)
    )

    @test eltype(plds_int.state_model.A) === Int
    @test eltype(plds_int.state_model.Q) === Int
    @test eltype(plds_int.state_model.x0) === Int
    @test eltype(plds_int.state_model.P0) === Int
    @test eltype(plds_int.obs_model.C) === Int
    @test eltype(plds_int.obs_model.log_d) === Int
    @test plds_int.latent_dim  == 2
    @test plds_int.obs_dim == 2

    # Float32
    A_f32 = Float32[1 2; 3 4]
    C_f32 = Float32[1 0; 0 1]
    Q_f32 = Float32[2 0; 0 2]
    log_d_f32 = Float32[0.5, 1.5]
    x0_f32 = Float32[0.7, 0.8]
    P0_f32 = Float32[4 0; 0 4]

    gsm_f32 = GaussianStateModel(A=A_f32, Q=Q_f32, x0=x0_f32, P0=P0_f32)
    pom_f32 = PoissonObservationModel(C=C_f32, log_d=log_d_f32)
    plds_f32 = LinearDynamicalSystem(
        state_model=gsm_f32,
        obs_model=pom_f32,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6)
    )

    @test eltype(plds_f32.state_model.A) === Float32
    @test eltype(plds_f32.obs_model.C) === Float32
    @test eltype(plds_f32.obs_model.log_d) === Float32

    # BigFloat
    A_bf = BigFloat[1 2; 3 4]
    C_bf = BigFloat[1 0; 0 1]
    Q_bf = BigFloat[2 0; 0 2]
    log_d_bf = BigFloat[0.1, 0.2]
    x0_bf = BigFloat[0.3, 0.4]
    P0_bf = BigFloat[4 0; 0 4]

    gsm_bf = GaussianStateModel(A=A_bf, Q=Q_bf, x0=x0_bf, P0=P0_bf)
    pom_bf = PoissonObservationModel(C=C_bf, log_d=log_d_bf)
    plds_bf = LinearDynamicalSystem(
        state_model=gsm_bf,
        obs_model=pom_bf,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6)
    )

    @test eltype(plds_bf.state_model.A) === BigFloat
    @test eltype(plds_bf.obs_model.C) === BigFloat
    @test eltype(plds_bf.obs_model.log_d) === BigFloat
end

function test_poisson_sample_type_preservation()
    # Float32
    A_f32 = Matrix{Float32}(I, 2, 2)
    C_f32 = Matrix{Float32}(I, 2, 2)
    Q_f32 = Matrix{Float32}(I, 2, 2)
    log_d32 = zeros(Float32, 2)
    x0_f32 = fill(one(Float32), 2)
    P0_f32 = Matrix{Float32}(I, 2, 2)

    gsm_f32 = GaussianStateModel(A=A_f32, Q=Q_f32, x0=x0_f32, P0=P0_f32)
    pom_f32 = PoissonObservationModel(C=C_f32, log_d=log_d32)
    plds_f32 = LinearDynamicalSystem(
        state_model=gsm_f32,
        obs_model=pom_f32,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6)
    )

    x_f32, y_f32 = rand(plds_f32; tsteps=50, ntrials=3)

    @test eltype(x_f32) === Float32
    @test eltype(y_f32) === Float32
    @test size(x_f32) == (2, 50, 3)
    @test size(y_f32) == (2, 50, 3)

    # BigFloat
    A_bf = Matrix{BigFloat}(I, 2, 2)
    C_bf = Matrix{BigFloat}(I, 2, 2)
    Q_bf = Matrix{BigFloat}(I, 2, 2)
    log_bf = zeros(BigFloat, 2)
    x0_bf = fill(one(BigFloat), 2)
    P0_bf = Matrix{BigFloat}(I, 2, 2)

    gsm_bf = GaussianStateModel(A=A_bf, Q=Q_bf, x0=x0_bf, P0=P0_bf)
    pom_bf = PoissonObservationModel(C=C_bf, log_d=log_bf)
    plds_bf = LinearDynamicalSystem(
        state_model=gsm_bf,
        obs_model=pom_bf,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 6)
    )

    x_bf, y_bf = rand(plds_bf; tsteps=50, ntrials=3)

    @test eltype(x_bf) === BigFloat
    @test eltype(y_bf) === BigFloat
    @test size(x_bf) == (2, 50, 3)
    @test size(y_bf) == (2, 50, 3)
end

function test_poisson_fit_type_preservation()
    for T in CHECKED_TYPES
        A     = Matrix{T}(I, 2, 2)
        C     = Matrix{T}(I, 2, 2)
        Q     = Matrix{T}(I, 2, 2)
        log_d = zeros(T, 2)
        x0    = fill(one(T), 2)
        P0    = Matrix{T}(I, 2, 2)

        gsm = GaussianStateModel(A=A, Q=Q, x0=x0, P0=P0)
        pom = PoissonObservationModel(C=C, log_d=log_d)
        lds = LinearDynamicalSystem(
            state_model=gsm,
            obs_model=pom,
            latent_dim=2,
            obs_dim=2,
            fit_bool=fill(true, 6)
        )
        
        x, y = rand(lds; tsteps=50, ntrials=3)

        mls, param_diff = fit!(lds, y; max_iter = 10, tol = 1e-6)

        @test eltype(mls) === T
        @test eltype(param_diff) === T
    end 
end 

function test_poisson_loglikelihood_type_preservation()
    for T in CHECKED_TYPES
        A     = Matrix{T}(I, 2, 2)
        C     = Matrix{T}(I, 2, 2)
        Q     = Matrix{T}(I, 2, 2)
        log_d = zeros(T, 2)
        x0    = fill(one(T), 2)
        P0    = Matrix{T}(I, 2, 2)

        gsm = GaussianStateModel(A=A, Q=Q, x0=x0, P0=P0)
        pom = PoissonObservationModel(C=C, log_d=log_d)
        lds = LinearDynamicalSystem(
            state_model=gsm,
            obs_model=pom,
            latent_dim=2,
            obs_dim=2,
            fit_bool=fill(true, 6)
        )

        x, y = rand(lds; tsteps=50, ntrials=3)

        x_mat = x[:, :, 1]  
        y_mat = y[:, :, 1]  

        ll = StateSpaceDynamics.loglikelihood(x_mat, lds, y_mat)

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
    @test poisson_lds.state_model.P0 == P0  # Fixed: was p0
    @test poisson_lds.obs_dim == 3
    @test poisson_lds.latent_dim == 2
    @test poisson_lds.fit_bool == [true, true, true, true, true, true]
end

function test_Gradient()
    plds, x, y = toy_PoissonLDS()

    # for each trial check the gradient
    for i in axes(y, 3)
        # numerically calculate the gradient
        f = latents -> StateSpaceDynamics.loglikelihood(latents, plds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x[:, :, i])

        # calculate the gradient
        grad = StateSpaceDynamics.Gradient(plds, y[:, :, i], x[:, :, i])

        @test norm(grad - grad_numerical) < 1e-8
    end
end

function test_Hessian()
    plds, x, y = toy_PoissonLDS()

    # create function that allows that we can pass the latents to the loglikelihood transposed
    function log_likelihood(x::AbstractArray, plds, y::AbstractArray)
        return StateSpaceDynamics.loglikelihood(x, plds, y)
    end

    # check hessian for each trial
    for i in axes(y, 3)
        hess, main, super, sub = StateSpaceDynamics.Hessian(
            plds, y[:, 1:3, i], x[:, 1:3, i]
        )
        @test size(hess) == (plds.latent_dim * 3, plds.latent_dim * 3)
        @test size(main) == (3,)
        @test size(super) == (2,)
        @test size(sub) == (2,)

        # calculate hess using autodiff now
        obj = latents -> log_likelihood(latents, plds, y[:, 1:3, i])
        hess_numerical = ForwardDiff.hessian(obj, x[:, 1:3, i])
        @test norm(hess_numerical - hess) < 1e-8
    end
end

function test_smooth()
    plds, x, y = toy_PoissonLDS()

    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, size(y, 2), size(y, 3))

    # smooth data
    ml_total = StateSpaceDynamics.smooth!(plds, tfs, y)

    nTrials = size(y, 3)
    nTsteps = size(y, 2)

    x_smooth = tfs[1].x_smooth
    p_smooth = tfs[1].p_smooth
    p_smooth_tt1 = tfs[1].p_smooth_tt1

    @test size(x_smooth) == (plds.latent_dim, nTsteps)
    @test size(p_smooth) == (plds.latent_dim, plds.latent_dim, nTsteps)
    @test size(p_smooth_tt1) == (plds.latent_dim, plds.latent_dim, nTsteps)

    # test gradient is zero
    for i in 1:nTrials
        # may as well test the gradient here too 
        f = latents -> StateSpaceDynamics.loglikelihood(latents, plds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x_smooth[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(plds, y[:, :, i], x_smooth[:, :, i])

        @test norm(grad_numerical - grad_analytical) < 1e-7
    end
end

function test_parameter_gradient()
    plds, x, y = toy_PoissonLDS()

    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, size(y, 2), size(y, 3))

    # run estep
    ml_total = StateSpaceDynamics.estep!(plds, tfs, y)

    # params
    C, log_d = plds.obs_model.C, plds.obs_model.log_d
    params = vcat(vec(C), log_d)
    
    # get analytical gradient
    grad_analytical = StateSpaceDynamics.gradient_observation_model!(
        zeros(length(params)), C, log_d, tfs, y
    )

    # get numerical gradient
    E_z = tfs[1].E_z
    P_smooth = tfs[1].p_smooth
    function f(params::AbstractVector{<:Real})
        C_size = plds.obs_dim * plds.latent_dim
        log_d = params[(end - plds.obs_dim + 1):end]
        C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
        return -StateSpaceDynamics.Q_observation_model(C, log_d, reshape(E_z, size(E_z)..., 1), reshape(P_smooth, size(P_smooth)..., 1), y)
    end

    grad = ForwardDiff.gradient(f, params)


    @test isapprox(grad, grad_analytical, rtol=1e-5, atol=1e-5)
end

function test_initial_observation_parameter_updates(ntrials::Int=1)
    plds, x, y = toy_PoissonLDS(ntrials, [true, true, false, false, false, false])

    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, size(y, 2), size(y, 3))

    # run estep
    ml_total = StateSpaceDynamics.estep!(plds, tfs, y)

    # optimize the x0 and p0 entries using autograd
    function obj(x0::AbstractVector, P0_sqrt::AbstractMatrix, plds)
        A, Q = plds.state_model.A, plds.state_model.Q
        P0 = P0_sqrt * P0_sqrt'
        Q_val = 0.0
        for i in 1:ntrials
            trial = tfs[i]
            Q_val += StateSpaceDynamics.Q_state(
                A, Q, P0, x0, trial.E_z, trial.E_zz, trial.E_zz_prev
            )
        end
        return -Q_val
    end

    P0_sqrt = Matrix(cholesky(plds.state_model.P0).U)

    x0_opt =
        optimize(
            x0 -> obj(x0, P0_sqrt, plds),
            plds.state_model.x0,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    P0_opt = optimize(P0_ -> obj(x0_opt, P0_, plds), P0_sqrt, LBFGS()).minimizer

    # update the initial state and covariance
    StateSpaceDynamics.mstep!(plds, tfs, y)

    @test isapprox(plds.state_model.x0, x0_opt, atol=1e-6)
    @test isapprox(plds.state_model.P0, P0_opt * P0_opt', atol=1e-6)
end

function test_state_model_parameter_updates(ntrials::Int=1)
    plds, x, y = toy_PoissonLDS(ntrials, [false, false, true, true, false, false])

    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, size(y, 2), size(y, 3))

    # run estep
    ml_total = StateSpaceDynamics.estep!(plds, tfs, y)

    # optimize the A and Q entries using autograd
    function obj(A::AbstractMatrix, Q_sqrt::AbstractMatrix, plds)
        Q = Q_sqrt * Q_sqrt'
        Q_val = 0.0
        for i in 1:ntrials
            trial = tfs[i]
            # calculate the Q value for this trial
            Q_val += StateSpaceDynamics.Q_state(
                A, Q, plds.state_model.P0, plds.state_model.x0,
                trial.E_z, trial.E_zz, trial.E_zz_prev
            )
        end
        return -Q_val
    end

    Q_sqrt = Matrix(cholesky(plds.state_model.Q).U)

    A_opt =
        optimize(
            A -> obj(A, Q_sqrt, plds),
            plds.state_model.A,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    Q_opt =
        optimize(
            Q_sqrt -> obj(A_opt, Q_sqrt, plds),
            Q_sqrt,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer

    # update the state model
    StateSpaceDynamics.mstep!(plds, tfs, y)

    @test isapprox(plds.state_model.A, A_opt, atol=1e-6)
    @test isapprox(plds.state_model.Q, Q_opt * Q_opt', atol=1e-6)
end

function test_EM(n_trials::Int=1)
    # generate fake data
    plds, x, y = toy_PoissonLDS(n_trials)

    # create a new plds model with random parameters
    A     = Matrix{Float64}(I, 2, 2)
    C     = Matrix{Float64}(I, 3, 2)  # Fixed dimensions for 3 obs, 2 latent
    Q     = Matrix{Float64}(I, 2, 2)
    log_d = zeros(Float64, 3)  # Fixed: 3 observations
    x0    = fill(one(Float64), 2)
    P0    = Matrix{Float64}(I, 2, 2)

    gsm_new = GaussianStateModel(A=A, Q=Q, x0=x0, P0=P0)
    pom_new = PoissonObservationModel(C=C, log_d=log_d)
    plds_new = LinearDynamicalSystem(
        state_model=gsm_new,
        obs_model=pom_new,
        latent_dim=2,
        obs_dim=3,
        fit_bool=fill(true, 6)
    )

    elbo, norm_grad = fit!(plds_new, y; max_iter=100)

    # check that the ELBO increases over the whole algorithm, we cannot use monotonicity as a check as we are using Laplace EM.
    @test elbo[end] > elbo[1]
end

function test_EM_matlab()
    # read data used to smooth the results
    data_1 = Matrix(CSV.read("test_data/trial1.csv", DataFrame))
    data_2 = Matrix(CSV.read("test_data/trial2.csv", DataFrame))
    data_3 = Matrix(CSV.read("test_data/trial3.csv", DataFrame))
    y = cat(data_1, data_2, data_3; dims=3)
    y = permutedims(y, [2, 1, 3])
    # read the matlab objects to compare results
    seq = matread("test_data/seq_matlab_3_trials_plds.mat")
    params = matread("test_data/params_matlab_3_trials_plds.mat")
    
    # create a new plds model using the new constructor pattern
    gsm = GaussianStateModel(
        A=[cos(0.1) -sin(0.1); sin(0.1) cos(0.1)],
        Q=0.00001 * Matrix{Float64}(I(2)),
        x0=[1.0, -1.0],
        P0=0.00001 * Matrix{Float64}(I(2))
    )
    
    pom = PoissonObservationModel(
        C=[1.2 1.2; 1.2 1.2; 1.2 1.2],
        log_d=log.([0.1, 0.1, 0.1])
    )
    
    plds = LinearDynamicalSystem(
        state_model=gsm,
        obs_model=pom,
        latent_dim=2,
        obs_dim=3,
        fit_bool=fill(true, 6)
    )

    tfs = StateSpaceDynamics.initialize_FilterSmooth(plds, size(y, 2), size(y, 3))


    # first smooth results
    ml_total = StateSpaceDynamics.estep!(plds, tfs, y)

    # check each E_z, E_zz, E_zz_prev are the sample
    for i in 1:3
        posterior_x = seq["seq"]["posterior"][i]["xsm"]
        posterior_cov = seq["seq"]["posterior"][i]["Vsm"]
        posterior_lagged_cov = seq["seq"]["posterior"][i]["VVsm"]

        @test isapprox(tfs[i].E_z, posterior_x, atol=1e-6)

        # TODO: Restructure matlab objects s.t. we can compare as below
        # @test isapprox(E_zz[:, :, :, i], posterior_cov, atol=1e-6)
        # @test isapprox(E_zz_prev[:, :, :, i], posterior_lagged_cov, atol=1e-6)
    end
    # now test the params
    fit!(plds, y; max_iter=1)
    params_obj = params["params"]["model"]
    @test isapprox(plds.state_model.A, params_obj["A"], atol=1e-5)
    @test isapprox(plds.state_model.Q, params_obj["Q"], atol=1e-5)
    @test isapprox(plds.obs_model.C, params_obj["C"], atol=1e-5)
    @test isapprox(plds.state_model.x0, params_obj["x0"], atol=1e-5)
    @test isapprox(plds.state_model.P0, params_obj["Q0"], atol=1e-5)
    @test isapprox(exp.(plds.obs_model.log_d), params_obj["d"], atol=1e-5)
end