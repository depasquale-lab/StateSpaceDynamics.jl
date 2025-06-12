# ## Simulating and Fitting a Linear Dynamical System

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to simulate a latent
# linear dynamical system and fit it using the EM algorithm.

# ## Load Packages

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using StableRNGs

#
rng = StableRNG(1234);

# ## Create a State-Space Model

obs_dim = 10
latent_dim = 2

A = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)]
Q = Matrix(0.1 * I(2))

x0 = [0.0; 0.0]
P0 = Matrix(0.1 * I(2))

C = randn(obs_dim, latent_dim)
R = Matrix(0.5 * I(obs_dim))

true_gaussian_sm = GaussianStateModel(;A=A, Q=Q, x0=x0, P0=P0)
true_gaussian_om = GaussianObservationModel(;C=C, R=R)
true_lds = LinearDynamicalSystem(;
    state_model=true_gaussian_sm,
    obs_model=true_gaussian_om,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)
)

# ## Simulate Latent and Observed Data

tSteps = 500
latents, observations = rand(rng, true_lds; tsteps=tSteps, ntrials=1)

# ## Plot Vector Field of Latent Dynamics

x = y = -3:0.5:3
X = repeat(x', length(y), 1)
Y = repeat(y, 1, length(x))

U = zeros(size(X))
V = zeros(size(Y))

for i in 1:size(X, 1)
    for j in 1:size(X, 2)
        v = A * [X[i,j], Y[i,j]]
        U[i,j] = v[1] - X[i,j]
        V[i,j] = v[2] - Y[i,j]
    end
end

magnitude = @. sqrt(U^2 + V^2)
U_norm = U ./ magnitude
V_norm = V ./ magnitude

p = quiver(X, Y, quiver=(U_norm, V_norm), color=:blue, alpha=0.3,
           linewidth=1, arrow=arrow(:closed, :head, 0.1, 0.1))
plot!(latents[1, :, 1], latents[2, :, 1], xlabel="x₁", ylabel="x₂",
      color=:black, linewidth=1.5, title="Latent Dynamics", legend=false)

# ## Plot Latent States and Observations

states = latents[:, :, 1]
emissions = observations[:, :, 1]

plot(size=(800, 600), layout=@layout[a{0.3h}; b])

lim_states = maximum(abs.(states))
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black,
          linewidth=2, label="", subplot=1)
end

plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), title="Simulated Latent States",
      yformatter=y->"", tickfontsize=12)

lim_emissions = maximum(abs.(emissions))
for n in 1:obs_dim
    plot!(1:tSteps, emissions[n, :] .- lim_emissions * (n-1), color=:black,
          linewidth=2, label="", subplot=2)
end

plot!(subplot=2, yticks=(-lim_emissions .* (obs_dim-1:-1:0), [L"y_{%$n}" for n in 1:obs_dim]),
      xlabel="time", xlims=(0, tSteps), title="Simulated Emissions",
      yformatter=y->"", tickfontsize=12)

plot!(link=:x, size=(800, 600), left_margin=10Plots.mm)

# ## Initialize a Model and Perform Smoothing

A_init = random_rotation_matrix(2, rng)
Q_init = Matrix(0.1 * I(2))
C_init = randn(obs_dim, latent_dim)
R_init = Matrix(0.5 * I(obs_dim))
x0_init = zeros(latent_dim)
P0_init = Matrix(0.1 * I(latent_dim))

gaussian_sm_init = GaussianStateModel(;A=A_init, Q=Q_init, x0=x0_init, P0=P0_init)
gaussian_om_init = GaussianObservationModel(;C=C_init, R=R_init)

naive_ssm = LinearDynamicalSystem(;
    state_model=gaussian_sm_init,
    obs_model=gaussian_om_init,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)
)

x_smooth, _, _ = StateSpaceDynamics.smooth(naive_ssm, observations)

plot()
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black, linewidth=2, label="", subplot=1)
    plot!(1:tSteps, x_smooth[d, :, 1] .+ lim_states * (d-1), color=:firebrick, linewidth=2, label="", subplot=1)
end
plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), yformatter=y->"", tickfontsize=12,
      title="True vs. Predicted Latent States (Pre-EM)")

# ## Fit Model Using EM Algorithm

elbo, _ = fit!(naive_ssm, observations; max_iter=100, tol=1e-6)

x_smooth, _, _ = StateSpaceDynamics.smooth(naive_ssm, observations)

plot()
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black, linewidth=2, label="", subplot=1)
    plot!(1:tSteps, x_smooth[d, :, 1] .+ lim_states * (d-1), color=:firebrick, linewidth=2, label="", subplot=1)
end
plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), yformatter=y->"", tickfontsize=12,
      title="True vs. Predicted Latent States (Post-EM)")

# ## Confirm the model converges
plot(elbo, xlabel="iteration", ylabel="ELBO", title="ELBO (Marginal Loglikelihood)", legend=false)
