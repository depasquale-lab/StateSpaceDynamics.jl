# # Choosing Latent Dimensionality for Linear Dynamical Systems (LDS)
# One of the most critical decisions when fitting an LDS is selecting the latent dimensionality K.
# Cross-validation is the universal approach that works for ANY state-space model - Gaussian LDS,
# Poisson LDS, nonlinear SSMs, etc. This tutorial demonstrates robust CV-based model selection.

## Load Required Packages
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using Statistics
using StableRNGs
using Printf

# Fix RNG for reproducible results
rng = StableRNG(1234);

## Create a True Gaussian LDS System
# For demonstration, we'll create a ground truth LDS with K=4 latent dimensions.
# This system will exhibit interesting dynamics like oscillations and decay patterns.

K_true = 4  # True latent dimensionality
D = 10       # Observation dimensionality
T = 300;    # Number of time steps

# Create interesting dynamics: oscillating + decaying modes
θ = π/12  # Oscillation frequency
λ = 0.92  # Decay rate

true_A = [cos(θ) -sin(θ)  0.0    0.0;
          sin(θ)  cos(θ)  0.0    0.0;
          0.0     0.0     λ      0.0;
          0.0     0.0     0.0    0.85*λ];

true_Q = 0.05 * Matrix(I(K_true)); # Process noise covariance
true_b = zeros(K_true)

Random.seed!(rng, 42) # Observation matrix - each latent dimension affects multiple observations
true_C = randn(rng, D, K_true) * 0.6;
true_d = zeros(D)

true_R = 0.1 * Matrix(I(D)); # Observation noise covariance

true_μ0 = zeros(K_true) # Initial state parameters
true_Σ0 = 0.1 * Matrix(I(K_true));

true_lds = LinearDynamicalSystem(
    GaussianStateModel(true_A, true_Q, true_b, true_μ0, true_Σ0),
    GaussianObservationModel(true_C, true_R, true_d),
    K_true,
    D,
    fill(true, 6)
);

latent_states, observations = rand(rng, true_lds; tsteps=T, ntrials=1); # Generate ground truth data

# Visualize the true latent dynamics and observations
p1 = plot(layout=(2,2), size=(1000, 600))

plot!(1:T, latent_states[1, :], label="Latent 1 (cos)",
      linewidth=2, subplot=1, title="Oscillating Modes")
plot!(1:T, latent_states[2, :], label="Latent 2 (sin)",
      linewidth=2, subplot=1)

plot!(1:T, latent_states[3, :], label="Latent 3 (decay)",
      linewidth=2, subplot=2, title="Decaying Modes")
plot!(1:T, latent_states[4, :], label="Latent 4 (decay)",
      linewidth=2, subplot=2)

plot!(1:T, observations[1, :], label="Obs 1", alpha=0.7, subplot=3, title="Observations 1-3")
plot!(1:T, observations[2, :], label="Obs 2", alpha=0.7, subplot=3)
plot!(1:T, observations[3, :], label="Obs 3", alpha=0.7, subplot=3)
plot!(1:T, observations[4, :], label="Obs 4", alpha=0.7, subplot=4, title="Observations 4-6")
plot!(1:T, observations[5, :], label="Obs 5", alpha=0.7, subplot=4)
plot!(1:T, observations[6, :], label="Obs 6", alpha=0.7, subplot=4)

p1

## Prepare Data for Cross-Validation
y_data = reshape(observations, D, T, 1)  # (obs_dim, tsteps, ntrials)

## Cross-Validation Setup
K_candidates = 1:8  # Test latent dimensions from 1 to 8
n_folds = 5         # Number of CV folds
fold_size = T ÷ n_folds;

# Storage for CV results
cv_scores = zeros(length(K_candidates), n_folds)
cv_mean = zeros(length(K_candidates))
cv_std = zeros(length(K_candidates));

println("Starting Cross-Validation for Model Selection...")
println("="^60)

## Perform K-Fold Cross-Validation
for (k_idx, K) in enumerate(K_candidates)
    println("Testing K = $K...")
    
    fold_scores = zeros(n_folds)
    
    for fold in 1:n_folds
        val_start = (fold - 1) * fold_size + 1
        val_end = min(fold * fold_size, T)
        
        train_indices = vcat(1:(val_start-1), (val_end+1):T)
        val_indices = val_start:val_end
        
        y_train = y_data[:, train_indices, :]
        y_val = y_data[:, val_indices, :]
        

        A_init = 0.9 * Matrix(I(K)) + 0.1 * randn(rng, K, K)
        Q_init = 0.1 * Matrix(I(K))
        C_init = randn(rng, D, K) * 0.5
        R_init = 0.2 * Matrix(I(D))
        μ0_init = zeros(K)
        Σ0_init = 0.1 * Matrix(I(K))
        
        lds_candidate = LinearDynamicalSystem(
            GaussianStateModel(A_init, Q_init, true_b, μ0_init, Σ0_init),
            GaussianObservationModel(C_init, R_init, true_d),
            K,
            D,
            fill(true, 6)  # Fit all parameters
        )
        
        try
            lls, _ = fit!(lds_candidate, y_train; max_iter=200, tol=1e-6, progress=false);
            
            x_val, _ = smooth(lds_candidate, y_val[:, :, 1])
            val_ll = loglikelihood(x_val, lds_candidate, y_val[:, :, 1])
            
            fold_scores[fold] = val_ll / length(val_indices)  # Normalize by sequence length
            
        catch e
            println("  Warning: Fold $fold failed for K=$K: $e")
            fold_scores[fold] = -Inf
        end
    end
    
    cv_scores[k_idx, :] = fold_scores
    cv_mean[k_idx] = mean(fold_scores)
    cv_std[k_idx] = std(fold_scores)

    @printf("  K=%d: CV Score = %.3f ± %.3f\n", K, cv_mean[k_idx], cv_std[k_idx])
end

## Find Optimal K
best_k_idx = argmax(cv_mean)
best_K = K_candidates[best_k_idx]

println("\n" * "="^60)
println("CROSS-VALIDATION RESULTS:")
println("="^60)
@printf("True K: %d\n", K_true)
@printf("Best K: %d (CV Score: %.3f ± %.3f)\n", best_K, cv_mean[best_k_idx], cv_std[best_k_idx])
println()

p2 = plot(K_candidates, cv_mean, 
          yerr=cv_std,
          marker=:circle, 
          markersize=6,
          linewidth=2,
          xlabel="Latent Dimensionality (K)",
          ylabel="Cross-Validation Score",
          title="Model Selection via Cross-Validation",
          legend=false,
          size=(800, 500))

vline!([K_true], linestyle=:dash, color=:green, linewidth=2, 
       annotations=[(K_true, maximum(cv_mean)-20, "True K=$K_true", :green)])
vline!([best_K], linestyle=:dot, color=:red, linewidth=2,
       annotations=[(best_K, maximum(cv_mean)-30, "Selected K=$best_K", :red)])

p2

# Initialize final model
A_final = 0.9 * Matrix(I(best_K)) + 0.1 * randn(rng, best_K, best_K)
Q_final = 0.1 * Matrix(I(best_K))
b_final = zeros(best_K)
C_final = randn(rng, D, best_K) * 0.5
R_final = 0.2 * Matrix(I(D))
μ0_final = zeros(best_K)
Σ0_final = 0.1 * Matrix(I(best_K))

final_lds = LinearDynamicalSystem(
    GaussianStateModel(A_final, Q_final, b_final, μ0_final, Σ0_final),
    GaussianObservationModel(C_final, R_final, true_d),
    best_K,
    D,
    fill(true, 6)
)

# Fit on full dataset
final_lls, _ = fit!(final_lds, y_data; max_iter=500, tol=1e-8)

# Compare Learned vs True Dynamics
# Use the correct input format for smooth function (needs 3D array)
x_learned, P_learned = smooth(final_lds, y_data)

plt1 = plot(
    1:length(final_lls), final_lls,
    linewidth=2,
    xlabel="EM Iteration",
    ylabel="Log-Likelihood",
    title="Learning Curve (Final Model)"
)

n_plot = min(4, best_K, K_true)
colors = [:blue, :red, :green, :orange]

plt2 = plot(title="True vs Learned Latent Dynamics", xlabel="Time", ylabel="Latent State Value")
for i in 1:n_plot
    if i <= size(latent_states, 1)
        plot!(plt2, 1:T, latent_states[i, :],
              label="True Latent $i", color=colors[i],
              linestyle=:solid, linewidth=2)
    end
    if i <= size(x_learned, 1)
        plot!(plt2, 1:T, x_learned[i, :],
              label="Learned Latent $i", color=colors[i],
              linestyle=:dash, linewidth=2)
    end
end

p3 = plot(plt1, plt2, layout = @layout([a; b]), size=(1000,600))
p3


# Compute reconstruction error
# `x_learned` is now `(latent_dim, tsteps, 1)`, so we need to handle the singleton trial dimension
x_learned = x_learned[:, :, 1]

y_pred = final_lds.obs_model.C * x_learned
reconstruction_error = mean((observations - y_pred).^2)

@printf("Reconstruction MSE: %.6f\n", reconstruction_error)
