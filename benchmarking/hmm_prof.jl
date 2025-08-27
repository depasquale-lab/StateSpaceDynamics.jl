cd("C:\\Users\\zlosc\\Documents\\GitHub\\StateSpaceDynamics.jl")

using Pkg
Pkg.activate("./benchmarking")

using StateSpaceDynamics
using Profile
using LinearAlgebra
using Plots
using Random

const SSD = StateSpaceDynamics


# Define the output dimensionality of the HMM
output_dim = 1

# # Define the transition matrix and the initial state distribution
# A = [0.99 0.01; 0.05 0.95];
# πₖ = [0.5; 0.5]

A = rand(6,6)
A .= A ./ sum(A, dims=2)

πₖ = rand(6)
πₖ .= πₖ ./ sum(πₖ)


# Initialize the emission models
μ_1 = rand(output_dim)
Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

μ_2 = rand(output_dim)
Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)

emission_3 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=0.1 * Matrix{Float64}(I, output_dim, output_dim))
emission_4 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=1.0 * Matrix{Float64}(I, output_dim, output_dim))
emission_5 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=3.0 * Matrix{Float64}(I, output_dim, output_dim))
emission_6 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=2.7 * Matrix{Float64}(I, output_dim, output_dim))



# The general HMM constructor is used as follows
model = HiddenMarkovModel(K=6, B=[emission_1, emission_2, emission_3, emission_4, emission_5, emission_6], A=A, πₖ=πₖ)



# Define number of samples to generate 
n=1000

# Use the SSD sampling function and store the hidden states and emission data
true_labels, data = rand(model, n=n)

emission_1 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=0.001 * Matrix{Float64}(I, output_dim, output_dim))
emission_2 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=5.23 * Matrix{Float64}(I, output_dim, output_dim))
emission_3 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=9.1 * Matrix{Float64}(I, output_dim, output_dim))
emission_4 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=0.88 * Matrix{Float64}(I, output_dim, output_dim))
emission_5 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=2.1 * Matrix{Float64}(I, output_dim, output_dim))
emission_6 = GaussianEmission(output_dim=output_dim, μ=rand(output_dim), Σ=3.1 * Matrix{Float64}(I, output_dim, output_dim))

A = rand(6,6)
A .= A ./ sum(A, dims=2)

πₖ = rand(6)
πₖ .= πₖ ./ sum(πₖ)

test_model = HiddenMarkovModel(K=6, B=[emission_1, emission_2, emission_3, emission_4, emission_5, emission_6], A=A, πₖ=πₖ)

# Fit the HMM to the data using the SSD fit!() function
@profview fit!(test_model, data, max_iters=100, tol=1e-100)

# # Show that we can properly recover the true parameters
# println("Recovered Emission 1?: ", isapprox(test_model.B[1].μ, model.B[1].μ, atol=0.1) || isapprox(test_model.B[1].μ, model.B[2].μ, atol=0.1))
# println("Recovered Emission 2?: ",isapprox(test_model.B[2].μ, model.B[2].μ, atol=0.1) || isapprox(test_model.B[2].μ, model.B[1].μ, atol=0.1))

# plot(lls)
# title!("Log-likelihood over EM Iterations")
# xlabel!("EM Iteration")
# ylabel!("Log-Likelihood")