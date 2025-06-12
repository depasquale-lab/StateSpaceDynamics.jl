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

## Create a Gaussian generalized linear model-hidden Markov model (GLM-HMM)

````julia
emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

A = [0.99 0.01; 0.05 0.95]
πₖ = [0.8; 0.2]

true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])
````

## Sample from the GLM-HMM

````julia
n = 20000
Φ = randn(3, n)
true_labels, data = rand(true_model, Φ, n=n)
````

## Visualize the sampled dataset

````julia
gr()

colors = [:dodgerblue, :crimson]

scatter(Φ[1, :], vec(data);
    color = colors[true_labels],
    ms = 3,
    label = "",
    xlabel = "Input Feature 1",
    ylabel = "Output",
    title = "GLM-HMM Sampled Data"
)

xvals = range(minimum(Φ[1, :]), stop=maximum(Φ[1, :]), length=100)

β1 = emission_1.β[:, 1]
y_pred_1 = β1[1] .+ β1[2] .* xvals
plot!(xvals, y_pred_1;
    color = :dodgerblue,
    lw = 3,
    label = "State 1 regression",
    legend = :topright,
)

β2 = emission_2.β[:, 1]
y_pred_2 = β2[1] .+ β2[2] .* xvals
plot!(xvals, y_pred_2;
    color = :crimson,
    lw = 3,
    label = "State 2 regression",
    legend = :topright,
)
````

## Initialize and fit a new HMM to the sampled data

````julia
A = [0.8 0.2; 0.1 0.9]
πₖ = [0.6; 0.4]
emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])
lls = SSD.fit!(test_model, data, Φ)

plot(lls)
title!("Log-likelihood over EM Iterations")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")
````

## Visualize the emission model predictions

````julia
gr()

state_colors = [:dodgerblue, :crimson]
true_colors = [:green, :orange]
pred_colors = [:teal, :yellow]

scatter(Φ[1, :], vec(data);
    color = state_colors[true_labels],
    ms = 3,
    alpha = 1.0,
    label = "",
    xlabel = "Input Feature 1",
    ylabel = "Output",
    title = "True vs. Predicted Regressions"
)

xvals = range(minimum(Φ[1, :]), stop=maximum(Φ[1, :]), length=100)

β1_true = emission_1.β[:, 1]
y_true_1 = β1_true[1] .+ β1_true[2] .* xvals
plot!(xvals, y_true_1;
    color = true_colors[1],
    lw = 3,
    linestyle = :solid,
    label = "State 1 (true)"
)

β2_true = emission_2.β[:, 1]
y_true_2 = β2_true[1] .+ β2_true[2] .* xvals
plot!(xvals, y_true_2;
    color = true_colors[2],
    lw = 3,
    linestyle = :solid,
    label = "State 2 (true)"
)

β1_pred = test_model.B[1].β[:, 1]
y_pred_1 = β1_pred[1] .+ β1_pred[2] .* xvals
plot!(xvals, y_pred_1;
    color = pred_colors[1],
    lw = 3,
    linestyle = :dash,
    label = "State 1 (pred)"
)

β2_pred = test_model.B[2].β[:, 1]
y_pred_2 = β2_pred[1] .+ β2_pred[2] .* xvals
plot!(xvals, y_pred_2;
    color = pred_colors[2],
    lw = 3,
    linestyle = :dash,
    label = "State 2 (pred)"
)
````

## Visualize the latent state predictions using Viterbi

````julia
pred_labels= viterbi(test_model, data, Φ);

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
all_data = Vector{Matrix{Float64}}()
Φ_total = Vector{Matrix{Float64}}()

num_trials = 100
n=1000
all_true_labels = []

for i in 1:num_trials
    Φ = randn(3, n)
    true_labels, data = rand(true_model, Φ, n=n)
    push!(all_true_labels, true_labels)
    push!(all_data, data)
    push!(Φ_total, Φ)
end
````

## Fitting an HMM to multiple, independent trials of data

````julia
A = [0.8 0.2; 0.1 0.9]
πₖ = [0.6; 0.4]
emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

lls = SSD.fit!(test_model, all_data, Φ_total)

plot(lls)
title!("Log-likelihood over EM Iterations")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")
````

## Visualize latent state predictions for multiple trials of data using Viterbi

````julia
all_pred_labels_vec = viterbi(test_model, all_data, Φ_total)
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

