using LinearAlgebra
# using Plots
using Random
using StateSpaceDynamics
const SSD = StateSpaceDynamics

# Create Emission Models
emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3, 2, 2, 3], :, 1))
emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4, -2, 3, 2], :, 1))
# Create Switching Regression Model
true_model = SwitchingGaussianRegression(K=2, input_dim=3, output_dim=1, include_intercept=true)
# Plug in the emission models
true_model.B[1] = emission_1
true_model.B[2] = emission_2
# Sample from the model
n = 1000
Φ = randn(3, n)
true_labels, data = SSD.sample(true_model, Φ, n=n)
test_model = SSD.SwitchingGaussianRegression(K=2, input_dim=3, output_dim=1, include_intercept=true)
X=Φ
Y=data
lls = [-Inf]

SSD.fit!(test_model, data, Φ; max_iters=1)

@profview for _ in 1:1000
    model = deepcopy(test_model)
    SSD.fit!(model, data, Φ; max_iters=100)
end

