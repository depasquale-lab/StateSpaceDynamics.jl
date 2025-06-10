function test_AR_emission_initialization()
    model = AutoRegressionEmission(output_dim=2, order=1, include_intercept=false, β=[0.5 -0.2; 0.1 0.5], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);
    
    @test model.output_dim == 2
    @test model.order == 1
    @test model.include_intercept == false
    @test model.β == [0.5 -0.2; 0.1 0.5]
    @test model.Σ == [0.001 0.0; 0.0 0.001]
    @test model.λ == 0.0
end