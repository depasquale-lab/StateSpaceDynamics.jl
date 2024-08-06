function HiddenMarkovModel_Gaussian_simulation(time_steps::Int)

    emission_model1 = Gaussian(data_dim=2)
    emission_model2 = Gaussian(data_dim=2, μ=[10.0, -10.0])

    true_model = HiddenMarkovModel(K=2, B=[emission_model1, emission_model2])

    state_sequence, Y = SSM.sample(true_model, time_steps=time_steps)

    return true_model, state_sequence, Y
end

function test_HiddenMarkovModel_E_step()
    time_steps = 1000
    true_model, state_sequence, data... = HiddenMarkovModel_Gaussian_simulation(time_steps)
    Y = data[1]

    centroids = kmeanspp_initialization(Y, 2)


    emission_1 = Gaussian(data_dim=2, μ=centroids[:, 1])
    emission_2 = Gaussian(data_dim=2, μ=centroids[:, 2])

    est_model = HiddenMarkovModel(
        K=2, 
        B=[
            emission_1, 
            emission_2])

    γ, ξ, α, β = E_step(est_model, data)

    # test α
    @test size(α) == (size(Y, 1), est_model.K)

    # test β
    @test size(β) == (size(Y, 1), est_model.K)

    # test γ
    @test size(γ) == (size(Y, 1), est_model.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))

    # test ξ
    @test size(ξ) == (size(Y, 1) - 1, true_model.K, true_model.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_HiddenMarkovModel_fit()
    time_steps = 1000
    true_model, state_sequence, data... = HiddenMarkovModel_Gaussian_simulation(time_steps)
    Y = data[1]

    centroids = kmeanspp_initialization(Y, 2)


    emission_1 = Gaussian(data_dim=2, μ=centroids[:, 1])
    emission_2 = Gaussian(data_dim=2, μ=centroids[:, 2])

    est_model = HiddenMarkovModel(
        K=2, 
        B=[
            emission_1, 
            emission_2])

    
    SSM.fit!(est_model, Y)

    means = [true_model.B[i].μ for i in 1:2]
    covs = [true_model.B[i].Σ for i in 1:2]

    pred_means = [est_model.B[i].μ for i in 1:2]
    # 0.2 because it checks norms of the vectors
    @test sort(pred_means) ≈ sort(means) atol=0.2
    pred_covs = [est_model.B[i].Σ for i in 1:2]
    @test pred_covs ≈ covs atol=0.3
end 



# implement a recursive version
function concrete_subtypes(t::DataType, subtypes_list::Vector{DataType}=Vector{DataType}())
    if isconcretetype(t)
        push!(subtypes_list, t)
    end

    for subtype in subtypes(t)
        concrete_subtypes(subtype, subtypes_list)
    end
    return subtypes_list
end

function test_valid_emissions()
    # get all concrete emissions in valid_emission_models and store in a list
    concrete_emissions_list = []
    for emission_model in valid_emission_models
        subtypes_list = concrete_subtypes(emission_model)
        for subtype in subtypes_list
            push!(concrete_emissions_list, subtype)
        end
    end
    
    for emission in concrete_emissions_list
        # try to call test_<emission type>_valid_emission_model()
        try
            eval(Meta.parse("test_$(emission)_valid_emission_model()"))
        catch e
            @warn "test_$(emission)_valid_emission_model() not implemented"
        end
    end

    
end
