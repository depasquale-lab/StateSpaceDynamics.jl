### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

## import packages
begin
    # change this to your system path
    using Pkg

    # In the package manager, use "dev path\to\local\package\under\development"
    using SSM

    using LinearAlgebra
    using Distributions
    using Random
    using Plots
    using CSV
    using DataFrames
end

# Let's simulate observations from a Gaussian Mixture Model

Random.seed!(123)

k = 2
data_dim = 2
μ₁, μ₂ = [0.1, 0.3], [-1.2, 2.2]
Σ₁, Σ₂ = [1 0.2; 0.2 1], [1 0.9; 0.9 1]

gmmIdeal = SSM.GaussianMixtureModel(k, data_dim)
gmmIdeal.μₖ[1, :] = μ₁
gmmIdeal.μₖ[2, :] = μ₂
gmmIdeal.Σₖ[1] = Σ₁
gmmIdeal.Σₖ[2] = Σ₂

# Generate and save data
# data = SSM.sample(gmmIdeal, 1000)
# CSV.write("notebooks/gmm_data.csv", DataFrame(data, :auto))

data = CSV.read("notebooks/gmm_data.csv", DataFrame)
data = Matrix(data)

p = scatter(data[:, 1], data[:, 2]; label="Samples")
plot!(; title="Gaussian Mixture", xlabel="x₁", ylabel="x₂")

savefig("notebooks/gmm_data_plot.png")

gmmEstimate = SSM.GaussianMixtureModel(k, data_dim)

# now lets fit the model!
SSM.fit!(gmmEstimate, data)

# Component matching based on nearest mean vector, NOT using the Hungarian algorithm
function compare_gmms(gmmIdeal, gmmEstimate)
    k = gmmIdeal.k
    paired_j = Bool[false for _ in 1:k]  # Track which gmmEstimate components have been paired

    mean_diffs = zeros(Float64, k)
    cov_diffs = zeros(Float64, k)
    mixing_coeff_diffs = zeros(Float64, k)

    for i in 1:k
        best_match_idx = 0
        best_match_diff = Inf

        for j in 1:k
            if !paired_j[j]
                mean_diff = norm(gmmIdeal.μₖ[i, :] - gmmEstimate.μₖ[j, :])
                if mean_diff < best_match_diff
                    best_match_diff = mean_diff
                    best_match_idx = j
                end
            end
        end

        # Mark the best match as paired
        paired_j[best_match_idx] = true

        # Calculate differences for the best match
        mean_diffs[i] = best_match_diff
        cov_diffs[i] = norm(gmmIdeal.Σₖ[i] - gmmEstimate.Σₖ[best_match_idx])
        mixing_coeff_diffs[i] = abs(gmmIdeal.πₖ[i] - gmmEstimate.πₖ[best_match_idx])
    end

    return mean_diffs, cov_diffs, mixing_coeff_diffs
end

begin
    # Define a function to evaluate the multivariate Gaussian
    function evaluate_mvg(x, y, μ, Σ)
        pos = [x, y]
        return pdf(MvNormal(μ, Σ), pos)
    end

    # Generate a grid for the contour
    xmin, xmax = extrema(data[:, 1])
    ymin, ymax = extrema(data[:, 2])
    xrange = range(xmin; stop=xmax, length=100)
    yrange = range(ymin; stop=ymax, length=100)

    # Plot each Gaussian in the GaussianMixtureModel
    for k in 1:(gmmEstimate.k)
        Z = [
            evaluate_mvg(x, y, gmmEstimate.μₖ[k, :], gmmEstimate.Σₖ[k]) for x in xrange,
            y in yrange
        ]
        contour!(xrange, yrange, Z'; levels=10, linewidth=2, label="Gaussian Estimate $k")
    end
end

# Display the plot
display(p)
savefig("notebooks/gmm_model_plot.png")

# Assuming gmmIdeal and gmmEstimate are defined and fit appropriately,
# you can compare them using:
mean_diffs, cov_diffs, mixing_coeff_diffs = compare_gmms(gmmIdeal, gmmEstimate)

println("Mean differences: ", mean_diffs)
println("Covariance differences: ", cov_diffs)
println("Mixing coefficient differences: ", mixing_coeff_diffs)

# Wait until 'enter' on command line
readline()
