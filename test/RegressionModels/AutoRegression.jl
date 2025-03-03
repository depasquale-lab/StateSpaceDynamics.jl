function test_AR_emission_initialization()
    model = AutoRegressionEmission(output_dim=2, order=1, include_intercept=false, β=[0.8 -0.6; 0.6 0.8])
    
    @test model.output_dim == 2
    @test model.order == 1
    @test model.include_intercept == false
    @test model.β == [0.8 -0.6; 0.6 0.8]
end