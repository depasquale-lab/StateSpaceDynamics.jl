# # Poisson LDS
#
# A Poisson LDS keeps Gaussian latent dynamics but emits non-negative count
# observations through an exponential link. This pattern fits neural spike
# trains, customer arrivals, or any non-negative integer time series whose
# rate is driven by a hidden continuous process.

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using StableRNGs

rng = StableRNG(54321);

# ## Model
#
# ```math
# \begin{aligned}
#     x_{t+1}     &= A x_t + b + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, Q), \\
#     \lambda_t   &= \exp(C x_t + d),           & y_{t,i}       &\sim \mathrm{Poisson}(\lambda_{t,i}).
# \end{aligned}
# ```
#
# The latent dynamics are identical to the Gaussian LDS; only the observation
# layer changes. `d` sets the per-channel baseline log-rate and `C` says how
# strongly each latent dimension drives each channel.

obs_dim = 5
latent_dim = 2

A = 0.95 * [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
Q = Matrix(0.05 * I(latent_dim))
b = zeros(latent_dim)
x0 = zeros(latent_dim)
P0 = Matrix(0.05 * I(latent_dim))

C = 0.6 * randn(rng, obs_dim, latent_dim)
d = log.(fill(0.3, obs_dim))

state_model = GaussianStateModel(; A=A, b=b, Q=Q, x0=x0, P0=P0)
obs_model = PoissonObservationModel(; C=C, d=d)
true_plds = LinearDynamicalSystem(;
    state_model=state_model,
    obs_model=obs_model,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 5),
);

# ## Simulation

tsteps = 200
latents, observations = rand(rng, true_plds, tsteps);

# Continuous latents on top, spike rasters below.

p_traces = let
    lim_x = maximum(abs, latents)
    p = plot(size=(800, 600), layout=@layout[a{0.3h}; b])
    for d in 1:latent_dim
        plot!(p, 1:tsteps, latents[d, :] .+ lim_x * (d - 1);
            color=:black, linewidth=2, label="", subplot=1)
    end
    plot!(p; subplot=1, title="Latents",
        yticks=(lim_x .* (0:latent_dim - 1), [L"x_%$d" for d in 1:latent_dim]),
        xticks=[], yformatter=y -> "")
    colors = palette(:default, obs_dim)
    for n in 1:obs_dim
        spike_times = findall(>(0), observations[n, :])
        for t in spike_times
            plot!(p, [t, t], [n - 0.4, n + 0.4];
                color=colors[n], linewidth=1, label="", subplot=2)
        end
    end
    plot!(p; subplot=2, title="Spike raster", xlabel="time",
        yticks=(1:obs_dim, [L"y_{%$n}" for n in 1:obs_dim]),
        ylims=(0.5, obs_dim + 0.5), grid=false)
end

# ## Smoothing
#
# Because Poisson likelihoods break Gaussian conjugacy, the posterior over
# latents is no longer Gaussian and [`smooth`](@ref) uses a Laplace
# approximation — it finds the MAP latent trajectory by Newton's method and
# reports the posterior mean and Hessian-derived covariance at the mode.

naive_plds = LinearDynamicalSystem(;
    state_model=GaussianStateModel(;
        A=random_rotation_matrix(latent_dim, rng),
        Q=Matrix(0.05 * I(latent_dim)),
        b=zeros(latent_dim),
        x0=zeros(latent_dim),
        P0=Matrix(0.05 * I(latent_dim)),
    ),
    obs_model=PoissonObservationModel(;
        C=0.5 * randn(rng, obs_dim, latent_dim),
        d=log.(fill(0.3, obs_dim)),
    ),
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 5),
);

x_pre, _ = StateSpaceDynamics.smooth(naive_plds, observations);

# ## Learning
#
# [`fit!`](@ref) runs Laplace-EM. It's slower per iteration than the Gaussian
# version because each E-step solves a Newton optimisation. Fewer iterations
# usually suffice to see clear parameter recovery.

elbos = fit!(naive_plds, observations; max_iter=15, tol=1e-4);

x_post, _ = StateSpaceDynamics.smooth(naive_plds, observations);

# After EM, the smoothed latent trajectory should align with the true one up
# to a linear change of basis.

p_compare = let
    lim_x = maximum(abs, latents)
    p = plot()
    for d in 1:latent_dim
        plot!(p, 1:tsteps, latents[d, :] .+ lim_x * (d - 1);
            color=:black, linewidth=2,
            label=(d == 1 ? "true" : ""), alpha=0.8)
        plot!(p, 1:tsteps, x_post[d, :] .+ lim_x * (d - 1);
            color=:firebrick, linewidth=2,
            label=(d == 1 ? "post-EM" : ""), alpha=0.8)
    end
    plot!(p; title="True vs. recovered latents",
        yticks=(lim_x .* (0:latent_dim - 1), [L"x_%$d" for d in 1:latent_dim]),
        xlabel="time", yformatter=y -> "", legend=:topright)
end

# ELBO trajectory. Convergence is less smooth than the Gaussian case because
# each E-step is itself an iterative inner solve.

p_elbo = plot(elbos; xlabel="iteration", ylabel="ELBO",
    legend=false, linewidth=2, marker=:circle, markersize=3,
    color=:darkgreen, title="Laplace-EM convergence")

# ## Tests  #src

using SSDTest  #src
using Test  #src

test_em_improves(elbos)  #src
test_lds_dimensions(true_plds; latent_dim=latent_dim, obs_dim=obs_dim)  #src
test_lds_dimensions(naive_plds; latent_dim=latent_dim, obs_dim=obs_dim)  #src
test_smooth_improves(latents, x_pre, x_post)  #src
@test all(>=(0), observations)  #src
