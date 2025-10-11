function test_pretty_printing()
    # Set up IO buffer for printing

    io1 = IOBuffer()
    objs = []

    # Gaussian Emission

    ge1 = GaussianEmission(5, rand(5), rand(5, 5))
    ge2 = GaussianEmission(2, rand(2), rand(2, 2))
    push!(objs, ge1, ge2)

    @test println(io1, ge1) === nothing
    @test println(io1, ge2) === nothing

    # Gaussian Regression Emission

    gre = GaussianRegressionEmission(3, 3, rand(3, 3), rand(3, 3), true, 0.5)
    push!(objs, gre)

    @test println(io1, gre) === nothing

    # AutoRegression Emission 

    are = AutoRegressionEmission(3, 3, gre)
    push!(objs, are)

    @test println(io1, are) === nothing

    # Bernoulli Regression Emission

    bre = BernoulliRegressionEmission(5, 5, rand(5, 5), false, 0.5)
    push!(objs, bre)

    @test println(io1, bre) === nothing

    # Poisson Regression Emission

    pre = PoissonRegressionEmission(5, 5, rand(5, 5), false, 0.5)
    push!(objs, pre)

    @test println(io1, pre) === nothing

    # Regression Optimization

    ro = StateSpaceDynamics.RegressionOptimization(
        pre, rand(2, 2), rand(2, 2), rand(2), (2, 2)
    )
    push!(objs, ro)

    @test println(io1, ro) === nothing

    # Forward Backward object

    fb = StateSpaceDynamics.ForwardBackward(
        rand(2, 2),
        rand(2, 2),
        rand(2, 2),
        rand(2, 2),
        rand(2, 2),  # ξ is 2D Matrix, not 3D!
    )
    push!(objs, fb)

    @test println(io1, fb) === nothing

    # Filter Smooth object
    fs = StateSpaceDynamics.FilterSmooth(
        rand(2, 2),      # x_smooth (2D)
        rand(2, 2, 2),   # p_smooth (3D)
        rand(2, 2, 2),   # p_smooth_tt1 (3D) 
        rand(2, 2),      # E_z (2D)
        rand(2, 2, 2),   # E_zz (3D)
        rand(2, 2, 2),   # E_zz_prev (3D)
        0.0,              # entropy (scalar)
    )
    push!(objs, fs)

    @test println(io1, fs) === nothing

    # Hidden Markov Model 

    hmm1 = HiddenMarkovModel(rand(5, 5), [gre, are, bre, pre, gre], rand(5), 5)
    hmm2 = HiddenMarkovModel(rand(2, 2), [gre, are], rand(2), 2)
    push!(objs, hmm1, hmm2)

    @test println(io1, hmm1) === nothing
    @test println(io1, hmm2) === nothing

    # Gaussian State Model 

    gsm1 = GaussianStateModel(rand(5, 5), rand(5, 5), rand(5), rand(5), rand(5, 5))
    gsm2 = GaussianStateModel(rand(2, 2), rand(2, 2), rand(2), rand(2), rand(2, 2))
    push!(objs, gsm1, gsm2)

    @test println(io1, gsm1) === nothing
    @test println(io1, gsm2) === nothing

    # Gaussian Observation Model 

    gom1 = GaussianObservationModel(rand(5, 5), rand(5, 5), rand(5))
    gom2 = GaussianObservationModel(rand(3, 3), rand(3, 3), rand(3))
    push!(objs, gom1, gom2)

    @test println(io1, gom1) === nothing
    @test println(io1, gom2) === nothing

    # Poisson Observation Model 

    pom1 = PoissonObservationModel(rand(5, 5), rand(5))
    pom2 = PoissonObservationModel(rand(2, 2), rand(2))
    push!(objs, pom1, pom2)

    @test println(io1, pom1) === nothing
    @test println(io1, pom2) === nothing

    # Linear Dynamical System 

    lds1 = LinearDynamicalSystem(gsm1, gom1, 5, 5, [true, true, true, true, true, true])
    lds2 = LinearDynamicalSystem(gsm2, pom2, 2, 2, [true, true, true, true, true])
    push!(objs, lds1, lds2)

    @test println(io1, lds1) === nothing
    @test println(io1, lds2) === nothing

    # Gaussian Mixture Model 

    gmm1 = GaussianMixtureModel(5, rand(5, 5), [rand(5, 5) for _ in 1:5], rand(5))
    gmm2 = GaussianMixtureModel(2, rand(2, 2), [rand(2, 2) for _ in 1:2], rand(2))
    push!(objs, gmm1, gmm2)

    @test println(io1, gmm1) === nothing
    @test println(io1, gmm2) === nothing

    # Poisson Mixture Model 

    pmm1 = PoissonMixtureModel(5, rand(5), rand(5))
    pmm2 = PoissonMixtureModel(2, rand(2), rand(2))
    push!(objs, pmm1, pmm2)

    @test println(io1, pmm1) === nothing
    @test println(io1, pmm2) === nothing

    # Probabilistic PCA

    ppca = ProbabilisticPCA(rand(5, 5), 0.5, rand(5))
    push!(objs, ppca)

    @test println(io1, ppca) === nothing

    # Switching Linear Dynamical System (SLDS)
    slds1 = SLDS(rand(5, 5), rand(5), [lds1, lds2, lds1, lds2, lds1])
    slds2 = SLDS(rand(2, 2), rand(2), [lds1, lds2])
    push!(objs, slds1, slds2)

    @test println(io1, slds1) === nothing
    @test println(io1, slds2) === nothing

    # testing `print_full`
    io2 = IOBuffer()

    for obj in objs
        @test print_full(io2, obj) === nothing
    end

    # last tests
    seekstart(io1)
    seekstart(io2)

    str1 = read(io1, String)
    str2 = read(io2, String)

    @test str1 isa String
    @test 5e3 < length(str1) < length(str2)

    return nothing
end
