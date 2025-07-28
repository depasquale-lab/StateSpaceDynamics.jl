function test_pretty_printing()
    # Set up IO buffer for printing
    
    io = IOBuffer()

    # Gaussian Emission

    ge1 = GaussianEmission(5, rand(5), rand(5,5))
    ge2 = GaussianEmission(2, rand(2), rand(2,2))

    @test println(io, ge1) === nothing
    @test println(io, ge2) === nothing

    # Gaussian Regression Emission

    gre = GaussianRegressionEmission(3, 3, rand(3,3), rand(3,3), true, 0.5)

    @test println(io, gre) === nothing

    # AutoRegression Emission 

    are = AutoRegressionEmission(3, 3, gre)

    @test println(io, are) === nothing

    # Bernoulli Regression Emission

    bre = BernoulliRegressionEmission(5, 5, rand(5,5), false, 0.5)

    @test println(io, bre) === nothing

    # Poisson Regression Emission

    pre = PoissonRegressionEmission(5, 5, rand(5,5), false, 0.5)
    
    @test println(io, pre) === nothing

    # Hidden Markov Model 

    hmm1 = HiddenMarkovModel(rand(5,5), [gre, are, bre, pre, gre], rand(5), 5)
    hmm2 = HiddenMarkovModel(rand(2,2), [gre, are], rand(2), 2)

    @test println(io, hmm1) === nothing
    @test println(io, hmm2) === nothing

    # Gaussian State Model 

    gsm1 = GaussianStateModel(rand(5,5), rand(5,5), rand(5), rand(5,5))
    gsm2 = GaussianStateModel(rand(2,2), rand(2,2), rand(2), rand(2,2))

    @test println(io, gsm1) === nothing
    @test println(io, gsm2) === nothing

    # Gaussian Observation Model 

    gom1 = GaussianObservationModel(rand(5,5), rand(5,5))
    gom2 = GaussianObservationModel(rand(3,3), rand(3,3))

    @test println(io, gom1) === nothing
    @test println(io, gom2) === nothing

    # Poisson Observation Model 

    pom1 = PoissonObservationModel(rand(5,5), rand(5))
    pom2 = PoissonObservationModel(rand(2,2), rand(2))

    @test println(io, pom1) === nothing
    @test println(io, pom2) === nothing

    # Linear Dynamical System 

    lds1 = LinearDynamicalSystem(gsm1, gom1, 5, 5, [true, true, true, true, true, true])
    lds2 = LinearDynamicalSystem(gsm2, pom2, 2, 2, [true, true, true, true, true])

    @test println(io, lds1) === nothing
    @test println(io, lds2) === nothing

    # Gaussian Mixture Model 

    gmm1 = GaussianMixtureModel(5, rand(5,5), [rand(5,5) for _ in 1:5], rand(5))
    gmm2 = GaussianMixtureModel(2, rand(2,2), [rand(2,2) for _ in 1:2], rand(2))

    @test println(io, gmm1) === nothing
    @test println(io, gmm2) === nothing

    # Poisson Mixture Model 

    pmm1 = PoissonMixtureModel(5, rand(5), rand(5))
    pmm2 = PoissonMixtureModel(2, rand(2), rand(2))

    @test println(io, pmm1) === nothing
    @test println(io, pmm2) === nothing

    # Probabalistic PCA

    ppca = ProbabilisticPCA(rand(5,5), 0.5, rand(5))

    @test println(io, ppca) === nothing

    # Switching Linear Dynamical System 

    slds1 = SwitchingLinearDynamicalSystem(rand(5,5), [lds1, lds2, lds1, lds2, lds1], rand(5), 5)
    slds2 = SwitchingLinearDynamicalSystem(rand(2,2), [lds1, lds2], rand(2), 2)

    @test println(io, slds1) === nothing
    @test println(io, slds2) === nothing

    # last two tests

    seekstart(io)
    
    str = read(io, String)

    @test str isa String
    @test length(str) > 5e3

    return nothing
end