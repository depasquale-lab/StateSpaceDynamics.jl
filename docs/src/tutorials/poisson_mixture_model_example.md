```@meta
EditURL = "../../examples/PoissonMixtureModel.jl"
```

## Simulating and Fitting a Poisson Mixture Model

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to
create a Poisson Mixture Model and fit it using the EM algorithm.

````@example poisson_mixture_model_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using StatsPlots
using Distributions

rng = StableRNG(1234);
nothing #hide
````

## Create a true PoissonMixtureModel to simulate from

````@example poisson_mixture_model_example
k = 3
true_λs = [5.0, 10.0, 25.0]  # Poisson means
true_πs = [0.25, 0.45, 0.3]   # Mixing weights

true_pmm = PoissonMixtureModel(k, true_λs, true_πs)
````

## Generate data from the true model

````@example poisson_mixture_model_example
n = 500
labels = rand(rng, Categorical(true_πs), n)
data = [rand(rng, Poisson(true_λs[labels[i]])) for i in 1:n]  # Vector{Int}
````

## Plot histogram of Poisson samples by component

````@example poisson_mixture_model_example
p1 = histogram(
    data;
    group=labels,
    bins=0:1:maximum(data),
    bar_position=:dodge,
    xlabel="Count",
    ylabel="Frequency",
    title="Poisson Mixture Samples by Component",
    alpha=0.7,
    legend=:topright,
)
p1
````

## Fit a new PoissonMixtureModel to the data

````@example poisson_mixture_model_example
fit_pmm = PoissonMixtureModel(k)
_, lls = fit!(fit_pmm, data; maxiter=100, tol=1e-6, initialize_kmeans=true)
````

## Plot log-likelihoods to visualize EM convergence

````@example poisson_mixture_model_example
p2 = plot(
    lls;
    xlabel="Iteration",
    ylabel="Log-Likelihood",
    title="EM Convergence (Poisson Mixture)",
    marker=:circle,
    label="log_likelihood",
)
p2
````

## Plot model PMFs over the data histogram

````@example poisson_mixture_model_example
p3 = histogram(
    data;
    bins=0:1:maximum(data),
    normalize=true,
    alpha=0.3,
    label="Data",
    xlabel="Count",
    ylabel="Density",
    title="Poisson Mixtures: Data and PMFs",
)

x = collect(0:maximum(data))
colors = [:red, :green, :blue]

for i in 1:k
    λi = fit_pmm.λₖ[i]
    πi = fit_pmm.πₖ[i]
    pmf_i = πi .* pdf.(Poisson(λi), x)
    plot!(
        p3, x, pmf_i;
        lw=2,
        c=colors[i],
        label="Comp $i (λ=$(round(λi, sigdigits=3)))",
    )
end

mix_pmf = reduce(+, (πi .* pdf.(Poisson(λi), x) for (λi, πi) in zip(fit_pmm.λₖ, fit_pmm.πₖ)))
plot!(
    p3, x, mix_pmf;
    lw=3, ls=:dash, c=:black,
    label="Mixture",
)

p3
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

