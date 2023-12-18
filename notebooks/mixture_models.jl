### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 371c7050-9bc4-11ee-205f-354baf99a94a
### import packages
begin
	# change this to your system path
	using Pkg
	Pkg.activate("/Users/ryansenne/Documents/Github/SSM")
	include("/Users/ryansenne/Documents/Github/SSM//src/SSM.jl")
	using .SSM
	using LinearAlgebra
	using Distributions
	using Random
	using Plots
end

# ╔═╡ 278317f2-36fe-4632-96d2-7b1d71190c53
using PlutoUI

# ╔═╡ daaed62c-9700-4098-a6c8-c5bf56dc7fc6
md"hi"

# ╔═╡ dd7fc95b-70a7-402c-a921-da9c1105dac8
# Let's simulate observations from a Gaussian Mixture Model
begin
	Random.seed!(123)
	# set means and covs
	μ₁, μ₂ = [0.1, 0.3], [-1.2, 2.2]
	Σ₁, Σ₂ = [1 0.2; 0.2 1], [1 0.9; 0.9 1]
	# simulate data
	data1, data2 = rand(MvNormal(μ₁, Σ₁), 500), rand(MvNormal(μ₂, Σ₂), 500)
	combined_data = Matrix{Float64}(hcat(data1, data2)')
	# create model and visualize data points
	gmm = SSM.GMM(2, 2, combined_data)
	scatter(data1[1, :], data1[2, :], label="k₁")
	scatter!(data2[1, :], data2[2, :], label="k₂")
	plot!(title="Gaussian Mixture", xlabel="x₁", ylabel="x₂")
end
	

# ╔═╡ 1466cf75-19a6-4835-a4d6-aaa57e6ad343
# now lets fit the model!
SSM.fit!(gmm, combined_data)

# ╔═╡ 2401f387-2e07-4180-ba15-c6afa204782e
begin
	# Define a function to evaluate the multivariate Gaussian
	function evaluate_mvg(x, y, μ, Σ)
    	pos = [x, y]
    	return pdf(MvNormal(μ, Σ), pos)
	end

	# Generate a grid for the contour
	xmin, xmax = extrema(combined_data[:, 1])
	ymin, ymax = extrema(combined_data[:, 2])
	xrange = range(xmin, stop=xmax, length=100)
	yrange = range(ymin, stop=ymax, length=100)

	# Plot each Gaussian in the GMM
	for k in 1:gmm.k_means
    	Z = [evaluate_mvg(x, y, gmm.μ_k[:, k], gmm.Σ_k[k]) for x in xrange, y in 			yrange]
    	contour!(xrange, yrange, Z', levels=10, linewidth=2, label="Gaussian $k")
	end

	# Show the plot
	plot!()
end

# ╔═╡ Cell order:
# ╠═278317f2-36fe-4632-96d2-7b1d71190c53
# ╠═daaed62c-9700-4098-a6c8-c5bf56dc7fc6
# ╠═371c7050-9bc4-11ee-205f-354baf99a94a
# ╠═dd7fc95b-70a7-402c-a921-da9c1105dac8
# ╠═1466cf75-19a6-4835-a4d6-aaa57e6ad343
# ╠═2401f387-2e07-4180-ba15-c6afa204782e
