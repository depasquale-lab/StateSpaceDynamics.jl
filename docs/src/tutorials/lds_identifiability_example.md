```@meta
EditURL = "../../examples/LDSIdentifiability.jl"
```

# Understanding Non-Identifiability in Linear Dynamical Systems

This tutorial walks through the fundamental non-identifiability issues in
Linear Dynamical Systems (LDS), shows them numerically, and adds **Procrustes alignment**
so we can compare models "apples to apples". It follows a simple pattern:
1) build a reference LDS and data; 2) generate *equivalent* models via similarity
   transforms; 3) show identical likelihood/predictions; 4) align states/parameters
   with Procrustes; 5) summarize diagnostics and discuss what *is* identifiable.

## Load Required Packages

````@example lds_identifiability_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using Statistics
using StableRNGs
using Printf

rng = StableRNG(12345);
nothing #hide
````

## Create a Reference ("True") LDS

````@example lds_identifiability_example
K_true = 3  # latent dimensionality
D      = 8  # observation dimensionality
T      = 200;  # time steps for training/demo
nothing #hide
````

Stable but nontrivial dynamics

````@example lds_identifiability_example
A_true = [0.9  0.1  0.0;
          -0.1 0.8  0.2;
           0.0 0.0  0.7];

Q_true = 0.05 * Matrix(I(K_true));
nothing #hide
````

Observation matrix with interpretable rows

````@example lds_identifiability_example
C_true = [1.0  0.5  0.0;   # Obs 1: mainly latent dim 1
          0.8  0.3  0.1;   # Obs 2: mix of dims 1 & 2
          0.2  1.0  0.0;   # Obs 3: mainly latent dim 2
          0.0  0.7  0.4;   # Obs 4: dims 2 & 3
          0.1  0.2  0.9;   # Obs 5: mainly latent dim 3
          0.3  0.0  0.8;   # Obs 6: dims 1 & 3
          0.6  0.4  0.2;   # Obs 7: mixture
          0.4  0.6  0.5]   # Obs 8: mixture

R_true  = 0.1 * Matrix(I(D))
x0_true = zeros(K_true)
P0_true = 0.2 * Matrix(I(K_true))

true_lds = LinearDynamicalSystem(
    GaussianStateModel(A_true, Q_true, x0_true, P0_true),
    GaussianObservationModel(C_true, R_true),
    K_true, D, fill(true, 6)
);
nothing #hide
````

Generate data from the reference model

````@example lds_identifiability_example
x_true, y_true = rand(rng, true_lds; tsteps=T, ntrials=1)

print("Generated data from reference LDS model
")
print("True latent dynamics eigenvalues: ", round.(eigvals(A_true), digits=3), "
")
print("Data variance explained by each latent dim: ", round.(var(x_true[:,:,1], dims=2)[:], digits=3), "
")
````

## Non-Identifiability: Similarity (Rotation) Invariance

For any invertible matrix R, the transformation produces an equivalent model:
  A' = R*A*R⁻¹,   C' = C*R⁻¹,   Q' = R*Q*Rᵀ,   x₀' = R*x₀,   P₀' = R*P₀*Rᵀ
Such models yield identical likelihoods and predictions.

A helper to build transformed copies

````@example lds_identifiability_example
function rotate_lds(lds, R)
    A_rot = R * lds.state_model.A * inv(R)
    Q_rot = R * lds.state_model.Q * R'
    C_rot = lds.obs_model.C * inv(R)
    x0_rot = R * lds.state_model.x0
    P0_rot = R * lds.state_model.P0 * R'
    return LinearDynamicalSystem(
        GaussianStateModel(A_rot, Q_rot, x0_rot, P0_rot),
        GaussianObservationModel(C_rot, lds.obs_model.R),
        size(A_rot, 1), size(C_rot, 1), fill(true, 6)
    )
end
````

A small mix of orthogonal and non-orthogonal transforms

````@example lds_identifiability_example
rotations = [
    [cos(π/4) -sin(π/4) 0.0;  sin(π/4) cos(π/4) 0.0;  0.0 0.0 1.0],     # R1: rot in (1,2)
    [1.0 0.0 0.0;              0.0 cos(π/2) -sin(π/2); 0.0 sin(π/2) cos(π/2)],  # R2: rot in (2,3) 90°
    Matrix(qr(randn(rng, K_true, K_true)).Q),                             # R3: random orthogonal
    [0.0 0.0 1.0; 0.0 1.0 0.0; 1.0 0.0 0.0],                              # R4: axis swap (1↔3)
    Diagonal([2.0, 0.5, -1.2]) |> Matrix,                                 # R5: scaling + sign flip
    [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]                               # R6: permutation (1↔2)
]

rot_names = [
    "R1: rot(1,2, 45°)",
    "R2: rot(2,3, 90°)",
    "R3: random orthogonal",
    "R4: axis swap (1↔3)",
    "R5: scaling+sign",
    "R6: permutation (1↔2)"
]
````

Helper diagnostics

````@example lds_identifiability_example
isorthogonal(R; atol=1e-10) = isapprox(R' * R, I(size(R,1)), atol=atol)

function subspace_angles_deg(C1, C2)
    Q1 = qr(C1).Q[:, 1:size(C1,2)]
    Q2 = qr(C2).Q[:, 1:size(C2,2)]
    σ = svdvals(Q1' * Q2)
    σ = clamp.(σ, -1.0, 1.0)           # numerical safety
    θ = acos.(σ)
    return θ .* (180/π)
end
````

Build rotated models

````@example lds_identifiability_example
rotated_models = [rotate_lds(true_lds, R) for R in rotations]
````

Compute likelihoods: should match (up to numerical tolerance)

````@example lds_identifiability_example
y_data = reshape(y_true, D, T, 1)
x_smooth_orig, _ = smooth(true_lds, y_data)
ll_orig = loglikelihood(x_smooth_orig[:,:,1], true_lds, y_true[:,:,1])

print("
" * "="^60 * "
")
print("ROTATION / SIMILARITY INVARIANCE DEMONSTRATION
")
print("="^60 * "
")
@printf("Original model likelihood: %.6f
", ll_orig)

for (name, R, model) in zip(rot_names, rotations, rotated_models)
    x_s_rot, _ = smooth(model, y_data)
    ll_rot = loglikelihood(x_s_rot[:,:,1], model, y_true[:,:,1])
    @printf("%-24s  LL: %.6f  ΔLL: %.3e  cond(R): %-8.2f  orth? %s
",
            name, ll_rot, abs(ll_rot - ll_orig), cond(R), isorthogonal(R) ? "yes" : "no")
end
````

## Visualize Parameter Differences (Before Alignment)

````@example lds_identifiability_example
p1 = plot(layout=(2,2), size=(1000, 800))
heatmap!(A_true, title="Original A", color=:RdBu, subplot=1, aspect_ratio=:equal)
heatmap!(rotated_models[3].state_model.A, title="Rotated A (R3)", color=:RdBu, subplot=2, aspect_ratio=:equal)
heatmap!(C_true, title="Original C", color=:RdBu, subplot=3, aspect_ratio=:equal)
heatmap!(rotated_models[3].obs_model.C, title="Rotated C (R3)", color=:RdBu, subplot=4, aspect_ratio=:equal)
````

## Procrustes Alignment: Apples-to-Apples Comparisons

The latent coordinates are arbitrary: we can rotate them without changing the model's
likelihood. To compare two fits (or a rotated copy) **fairly**, align the states with an
orthogonal Procrustes transform R̂ that minimizes ‖R̂ X - Y‖_F.

````@example lds_identifiability_example
function procrustes_R(X::AbstractMatrix, Y::AbstractMatrix; proper::Bool=false)
    S = svd(Y * X')
    R̂ = S.U * S.Vt                    # (not S.U * S.V!)
    if proper && det(R̂) < 0           # optionally enforce det=+1
        U2 = copy(S.U); U2[:,end] .= -U2[:,end]
        R̂ = U2 * S.Vt
    end
    return R̂
end
````

Choose R3 for demonstration (random orthogonal)

````@example lds_identifiability_example
R_idx = 3
m_rot = rotated_models[R_idx]
x_rot, _ = smooth(m_rot, y_data)

Rhat = procrustes_R(x_rot[:,:,1], x_smooth_orig[:,:,1])  # map rotated -> original
state_align_relerr = norm(Rhat * x_rot[:,:,1] - x_smooth_orig[:,:,1]) / norm(x_smooth_orig[:,:,1])
@printf("
Procrustes state alignment (R%d): rel. error = %.3e
", R_idx, state_align_relerr)
````

Align parameters via R̂ for direct visual comparison:
  Ã = R̂ * A_rot * R̂'     and     C̃ = C_rot * R̂'

````@example lds_identifiability_example
A_rot_aligned = Rhat * m_rot.state_model.A * Rhat'
C_rot_aligned = m_rot.obs_model.C * Rhat'

ΔA = norm(A_true - A_rot_aligned)
ΔC = norm(C_true - C_rot_aligned)
@printf("Aligned parameter diffs (R%d): ||ΔA||=%.3e  ||ΔC||=%.3e
", R_idx, ΔA, ΔC)
````

Visualize with matched color scales for fairness

````@example lds_identifiability_example
Amin = minimum([minimum(A_true), minimum(A_rot_aligned)])
Amax = maximum([maximum(A_true), maximum(A_rot_aligned)])
Cmin = minimum([minimum(C_true), minimum(C_rot_aligned)])
Cmax = maximum([maximum(C_true), maximum(C_rot_aligned)])

p_align = plot(layout=(2,2), size=(1000, 800))
heatmap!(A_true, title="Original A", color=:RdBu, subplot=1, aspect_ratio=:equal, clims=(Amin, Amax))
heatmap!(A_rot_aligned, title="Aligned A (R3)", color=:RdBu, subplot=2, aspect_ratio=:equal, clims=(Amin, Amax))
heatmap!(C_true, title="Original C", color=:RdBu, subplot=3, aspect_ratio=:equal, clims=(Cmin, Cmax))
heatmap!(C_rot_aligned, title="Aligned C (R3)", color=:RdBu, subplot=4, aspect_ratio=:equal, clims=(Cmin, Cmax))
````

Residual over time (after Procrustes): should be ~0 except numerical noise

````@example lds_identifiability_example
restit = [norm(Rhat * x_rot[:, t, 1] - x_smooth_orig[:, t, 1]) for t in 1:T]
plot(1:T, restit, lw=2, xlabel="time", ylabel="‖R̂ x_rot − x_orig‖₂",
     title="Procrustes residual over time (R3)")
````

## Invariants: What *is* identifiable?

Similarity transforms preserve certain summaries:
- eigenvalues of A (up to ordering), hence modal timescales τ ≈ -1/log|λ|
- column space of C (compare via principal angles)

````@example lds_identifiability_example
function invariants_summary(lds)
    λ = eigvals(lds.state_model.A)
    τ = [-1 / log(abs(l)) for l in λ]  # (real-mode heuristic)
    return λ, τ
end

λ_true, τ_true = invariants_summary(true_lds)

for (i, (name, model)) in enumerate(zip(rot_names, rotated_models))
    λ_i, τ_i = invariants_summary(model)
    θ = subspace_angles_deg(C_true, model.obs_model.C)
    @printf("Invariant check %-24s  max|Δλ|=%.2e  max|Δτ|=%.2e  max angle(C)=%.3f°
",
            name,
            maximum(abs.(sort(λ_true; by=abs) - sort(λ_i; by=abs))),
            maximum(abs.(sort(τ_true; by=abs) - sort(τ_i; by=abs))),
            maximum(θ))
end
````

## Observational Equivalence: Predictions Match

New data from a rotated model and prediction with both models should have identical error.

````@example lds_identifiability_example
x_rot_new, y_rot_new = rand(rng, rotated_models[1]; tsteps=100, ntrials=1)

test_data = reshape(y_rot_new, D, 100, 1)

x_pred_orig, _ = smooth(true_lds, test_data)
y_pred_orig = true_lds.obs_model.C * x_pred_orig[:, :, 1]

x_pred_rot, _ = smooth(rotated_models[1], test_data)
y_pred_rot = rotated_models[1].obs_model.C * x_pred_rot[:, :, 1]

mse_orig = mean((y_rot_new[:, :, 1] - y_pred_orig).^2)
mse_rot  = mean((y_rot_new[:, :, 1] - y_pred_rot).^2)
@printf("
Prediction MSE (same test seq): original=%.6f  rotated=%.6f
", mse_orig, mse_rot)
````

## Summary Diagnostics Across Transforms

A compact table: ΔLL, cond(R), orthogonality, max subspace angle for C, and
Procrustes alignment error versus the original smoothed states.

````@example lds_identifiability_example
struct RotDiag
    name::String
    dLL::Float64
    condR::Float64
    orth::Bool
    max_angle_deg::Float64
    proc_relerr::Float64
end

diagnostics = RotDiag[]

for (name, R, model) in zip(rot_names, rotations, rotated_models)
    x_s, _ = smooth(model, y_data)
    ll = loglikelihood(x_s[:,:,1], model, y_true[:,:,1])
    dLL = abs(ll - ll_orig)
    Rhat_i = procrustes_R(x_s[:,:,1], x_smooth_orig[:,:,1])
    relerr = norm(Rhat_i * x_s[:,:,1] - x_smooth_orig[:,:,1]) / max(norm(x_smooth_orig[:,:,1]), eps())
    θ = subspace_angles_deg(C_true, model.obs_model.C)
    push!(diagnostics, RotDiag(name, dLL, cond(R), isorthogonal(R), maximum(θ), relerr))
end

print("
" * "-"^90 * "
")
@printf("%-24s | %-9s | %-8s | %-6s | %-14s | %-14s
",
        "Transform", "ΔLL", "cond(R)", "orth?", "max angle(C)°", "Procrustes err")
print("-"^90 * "
")
for d in diagnostics
    @printf("%-24s | %-.3e | %-8.2f | %-6s | %-14.3f | %-14.3e
",
            d.name, d.dLL, d.condR, d.orth ? "yes" : "no", d.max_angle_deg, d.proc_relerr)
end
print("-"^90 * "
")
````

## Practical Takeaways

- Don’t interpret individual latent coordinates; they are defined only up to an invertible change of basis.
- When comparing models/fits, either **align** with Procrustes or **report invariants** (eigenvalues/timescales, subspace angles, predictive metrics).
- Watch conditioning: extreme transforms (large cond(R)) can inflate numerical errors even when theory says models are identical.
- (Optional next step) Implement a small **canonicalization**: whiten Q≈I, real-Schur form for A, mode ordering & sign conventions. This “fixes the gauge” so different runs are directly comparable without per-pair alignment.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

