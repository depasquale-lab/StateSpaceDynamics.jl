```@meta
EditURL = "../../examples/PoissonLDS.jl"
```

# Simulating and Fitting a Poisson Linear Dynamical System

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to simulate and fit a
Linear Dynamical System (LDS) with Poisson observations using the Laplace-EM algorithm.

## Load Packages

````@example poisson_latent_dynamics_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
````

## Create a Poisson Linear Dynamical System

````@example poisson_latent_dynamics_example
obs_dim = 10
latent_dim = 2
````

State model parameters

````@example poisson_latent_dynamics_example
A = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)]
Q = Matrix(0.1 * I(latent_dim))
x0 = zeros(latent_dim)
P0 = Matrix(0.1 * I(latent_dim))
````

Observation model parameters

````@example poisson_latent_dynamics_example
log_d = log.(fill(0.1, obs_dim))
C = permutedims([abs.(randn(obs_dim))'; abs.(randn(obs_dim))'])

state_model = GaussianStateModel(; A, Q, x0, P0)
obs_model = PoissonObservationModel(; C, log_d)

true_plds = LinearDynamicalSystem(;
    state_model=state_model,
    obs_model=obs_model,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)
)
````

## Simulate Latent States and Observations

````@example poisson_latent_dynamics_example
tSteps = 500
latents, observations = rand(true_plds; tsteps=tSteps, ntrials=1)
````

## Plot Vector Field of Latent Dynamics

````@example poisson_latent_dynamics_example
x = y = -3:0.5:3
X = repeat(x', length(y), 1)
Y = repeat(y, 1, length(x))
U = zeros(size(X))
V = zeros(size(Y))

for i in 1:size(X, 1), j in 1:size(X, 2)
    v = A * [X[i,j], Y[i,j]]
    U[i,j] = v[1] - X[i,j]
    V[i,j] = v[2] - Y[i,j]
end

magnitude = @. sqrt(U^2 + V^2)
U_norm = U ./ magnitude
V_norm = V ./ magnitude

p = quiver(X, Y, quiver=(U_norm, V_norm), color=:blue, alpha=0.3,
           linewidth=1, arrow=arrow(:closed, :head, 0.1, 0.1))
plot!(latents[1, :, 1], latents[2, :, 1], xlabel="x₁", ylabel="x₂",
      color=:black, linewidth=1.5, title="Latent Dynamics", legend=false)
````

## Plot Latent States and Observations

````@example poisson_latent_dynamics_example
states = latents[:, :, 1]
emissions = observations[:, :, 1]
time_bins = size(states, 2)

plot(size=(800, 600), layout=@layout[a{0.3h}; b])
````

Latent states

````@example poisson_latent_dynamics_example
lim_states = maximum(abs.(states))
for d in 1:latent_dim
    plot!(1:time_bins, states[d, :] .+ lim_states * (d-1), color=:black,
          linewidth=2, label="", subplot=1)
end

plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, time_bins), title="Simulated Latent States",
      yformatter=y->"", tickfontsize=12)
````

Spiking observations

````@example poisson_latent_dynamics_example
colors = palette(:default, obs_dim)
for f in 1:obs_dim
    spike_times = findall(x -> x > 0, emissions[f, :])
    for t in spike_times
        plot!([t, t], [f-0.4, f+0.4], color=colors[f], linewidth=1, label="", subplot=2)
    end
end

plot!(subplot=2, yticks=(1:obs_dim, [L"y_{%$d}" for d in 1:obs_dim]),
      xlims=(0, time_bins), ylims=(0.5, obs_dim + 0.5), title="Simulated Emissions",
      xlabel="Time", tickfontsize=12, grid=false)
````

## Initialize Model and Smooth

Initialize with random parameters

````@example poisson_latent_dynamics_example
A_init = random_rotation_matrix(latent_dim)
Q_init = Matrix(0.1 * I(latent_dim))
C_init = randn(obs_dim, latent_dim)
log_d_init = log.(fill(0.1, obs_dim))
x0_init = zeros(latent_dim)
P0_init = Matrix(0.1 * I(latent_dim))

sm_init = GaussianStateModel(; A=A_init, Q=Q_init, x0=x0_init, P0=P0_init)
om_init = PoissonObservationModel(; C=C_init, log_d=log_d_init)

naive_plds = LinearDynamicalSystem(;
    state_model=sm_init,
    obs_model=om_init,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)
)

smoothed_x, smoothed_p, _ = smooth(naive_plds, observations)

plot()
for d in 1:latent_dim
    plot!(1:time_bins, states[d, :] .+ lim_states * (d-1), color=:black, linewidth=2, label="", subplot=1)
    plot!(1:time_bins, smoothed_x[d, :, 1] .+ lim_states * (d-1), color=:red, linewidth=2, label="", subplot=1)
end

plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, time_bins), title="True vs. Predicted Latent States (Pre-EM)",
      yformatter=y->"", tickfontsize=12)
````

## Fit the Poisson LDS Using Laplace EM

````@example poisson_latent_dynamics_example
elbo, _ = fit!(naive_plds, observations; max_iter=25, tol=1e-6)

smoothed_x, smoothed_p, _ = smooth(naive_plds, observations)

plot()
for d in 1:latent_dim
    plot!(1:time_bins, states[d, :] .+ lim_states * (d-1), color=:black, linewidth=2, label="", subplot=1)
    plot!(1:time_bins, smoothed_x[d, :, 1] .+ lim_states * (d-1), color=:red, linewidth=2, label="", subplot=1)
end

plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, time_bins), title="True vs. Predicted Latent States (Post-EM)",
      yformatter=y->"", tickfontsize=12)
````

## ELBO Convergence

````@example poisson_latent_dynamics_example
plot(elbo, xlabel="iteration", ylabel="ELBO", title="ELBO over Iterations", legend=false)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

