### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ b13d0a8e-af31-11ee-2929-cb6bddd69fbf
begin
	using Pkg
	Pkg.activate("/Users/ryansenne/Documents/GitHub/SSM")
	using Random
	using LinearAlgebra
	using Distributions
	using Plots
	include("/Users/ryansenne/Documents/GitHub/SSM/src/SSM.jl")
	using .SSM
end

# ╔═╡ a3f21cb8-335c-44ce-bf85-e17db819fe1b
# simulate data from a PPCA model that we can then learn the parameters
begin
	# set constants
	lat_dim = 2
	obs_dim = 4
	n = 1000
	W = randn(obs_dim, lat_dim)
	noise_var = 0.3
	# generate data from the latent space
	Z = randn(n, lat_dim)
	# simulate data based on parameters
	X = Z * W' + (randn(n, obs_dim) * sqrt(noise_var))
	# plot the first 3 dimensions of X
	Plots.scatter(X[1:end, 1], X[1:end, 2], X[1:end, 3])
end

# ╔═╡ 43a4e1fc-1a66-47f3-8063-2aea7bade03c
# now we want to learn the underlying parameters of the model using PPCA
begin
	# instantiate the model
	ppca = SSM.PPCA(X, 2)
	# now we can fit the model using the EM algorithm
	SSM.fit!(ppca, X)
end

# ╔═╡ 709b8649-f2a6-4643-9992-c0ba6027172a
W

# ╔═╡ 404a5042-6faa-4644-81c7-b4c59d10e1d4
ppca.W

# ╔═╡ 89292b91-e09b-486d-a9b0-e7f38b434335
norm(ppca.W[:, 1])

# ╔═╡ d968529f-d67f-472b-803b-ce9f791a107e
# plot the results of the PPCA model
scatter(ppca.z[:, 1], ppca.z[:, 2])

# ╔═╡ Cell order:
# ╠═b13d0a8e-af31-11ee-2929-cb6bddd69fbf
# ╠═a3f21cb8-335c-44ce-bf85-e17db819fe1b
# ╠═43a4e1fc-1a66-47f3-8063-2aea7bade03c
# ╠═709b8649-f2a6-4643-9992-c0ba6027172a
# ╠═404a5042-6faa-4644-81c7-b4c59d10e1d4
# ╠═89292b91-e09b-486d-a9b0-e7f38b434335
# ╠═d968529f-d67f-472b-803b-ce9f791a107e
