## Simulating and Fitting a Hidden Markov Model

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create, sample from, and fit Hidden
Markov Models (HMMs).

## Load Packages

````julia
using LinearAlgebra
using Plots
using Random
using StateSpaceDynamics
````

````julia
rng = StableRNG(1234);
````

## Create an HMM

````julia
output_dim = 2

A = [0.99 0.01; 0.05 0.95];
πₖ = [0.5; 0.5]

μ_1 = [-1.0, -1.0]
Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

μ_2 = [1.0, 1.0]
Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)

model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)
````

## Sample from the HMM

````julia
num_samples = 10000
true_labels, data = rand(model, n=num_samples)
````

## Visualize the sampled dataset

````julia
gr()

x_vals = data[1, 1:num_points]
y_vals = data[2, 1:num_points]
labels_slice = true_labels[1:num_points]

state_colors = [:dodgerblue, :crimson]

plt = plot()
for state in 1:2
    idx = findall(labels_slice .== state)
    scatter!(x_vals[idx], y_vals[idx];
        color=state_colors[state],
        label="State $state",
        markersize=6)
end

plot!(x_vals, y_vals;
    color=:gray,
    lw=1.5,
    linealpha=0.4,
    label="")

scatter!([x_vals[1]], [y_vals[1]];
    color=:green,
    markershape=:star5,
    markersize=10,
    label="Start")

scatter!([x_vals[end]], [y_vals[end]];
    color=:black,
    markershape=:diamond,
    markersize=8,
    label="End")

xlabel!("Output dim 1")
ylabel!("Output dim 2")
title!("Emissions from HMM (First 100 Points)")
````

## Initialize and fit a new HMM to the sampled data

````julia
μ_1 = [-0.25, -0.25]
Σ_1 = 0.3 * Matrix{Float64}(I, output_dim, output_dim)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

μ_2 = [0.25, 0.25]
Σ_2 = 0.5 * Matrix{Float64}(I, output_dim, output_dim)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

A = [0.8 0.2; 0.05 0.95]
πₖ = [0.6,0.4]
test_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

lls = fit!(test_model, data)

plot(lls)
title!("Log-likelihood over EM Iterations")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")
````

## Visualize the latent state predictions using Viterbi

````julia
pred_labels= viterbi(test_model, data);

gr()

true_mat = reshape(true_labels[1:1000], 1, :)
pred_mat = reshape(pred_labels[1:1000], 1, :)

p1 = heatmap(true_mat;
    colormap = :roma50,
    title = "True State Labels",
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    colorbar = false,
    framestyle = :box)

p2 = heatmap(pred_mat;
    colormap = :roma50,
    title = "Predicted State Labels",
    xlabel = "Timepoints",
    ylabel = "",
    xticks = 0:200:1000,
    yticks = false,
    colorbar = false,
    framestyle = :box)

plot(p1, p2;
    layout = (2, 1),
    size = (700, 500),
    margin = 5Plots.mm)
````

## Sampling multiple, independent trials of data from an HMM

````julia
n_trials = 100
n_samples = 1000

all_true_labels = Vector{Vector{Int}}(undef, n_trials)
all_data = Vector{Matrix{Float64}}(undef, n_trials)

for i in 1:n_trials
    true_labels, data = rand(true_model, n=n_samples)
    all_true_labels[i] = true_labels
    all_data[i] = data
end
````

## Fitting an HMM to multiple, independent trials of data

````julia
μ_1 = [-0.25, -0.25]
Σ_1 = 0.3 * Matrix{Float64}(I, output_dim, output_dim)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

μ_2 = [0.25, 0.25]
Σ_2 = 0.5 * Matrix{Float64}(I, output_dim, output_dim)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

A = [0.8 0.2; 0.05 0.95]
πₖ = [0.6,0.4]
test_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

lls = SSD.fit!(test_model, all_data)

plot(lls)
title!("Log-likelihood over EM Iterations")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")
````

## Visualize latent state predictions for multiple trials of data using Viterbi

````julia
all_pred_labels_vec = viterbi(test_model, all_data)
all_pred_labels = hcat(all_pred_labels_vec...)'
all_true_labels_matrix = hcat(all_true_labels...)'

gr()
state_colors = [:dodgerblue, :crimson]
true_subset = all_true_labels_matrix[1:10, 1:500]
pred_subset = all_pred_labels[1:10, 1:500]

p1 = heatmap(
    true_subset,
    colormap = :roma50,
    colorbar = false,
    title = "True State Labels",
    xlabel = "",
    ylabel = "Trials",
    xticks = false,
    yticks = true,
    margin = 5Plots.mm,
    legend = false
)

p2 = heatmap(
    pred_subset,
    colormap = :roma50,
    colorbar = false,
    title = "Predicted State Labels",
    xlabel = "Timepoints",
    ylabel = "Trials",
    xticks = true,
    yticks = true,
    margin = 5Plots.mm,
    legend = false
)

final_plot = plot(
    p1, p2,
    layout = (2, 1),
    size = (850, 550),
    margin = 5Plots.mm,
    legend = false,
)

display(final_plot)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

