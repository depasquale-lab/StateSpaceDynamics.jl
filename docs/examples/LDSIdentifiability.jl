# # Non-identifiability of LDS coordinates
#
# An LDS's latent coordinates are identifiable only up to an invertible
# change of basis. For any invertible $R$, the transformed model
# $(A', Q', C', x_0', P_0') = (R A R^{-1},\, R Q R^\top,\, C R^{-1},\, R x_0,\, R P_0 R^\top)$
# is observationally equivalent: same likelihood, same predictions. Here we
# demonstrate the equivalence numerically and show how Procrustes alignment
# lets us compare fits "apples-to-apples".

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using Statistics
using StableRNGs
using Printf

rng = StableRNG(12345);

# ## Reference model

K_true = 3
D = 8
T = 200

A_true = [0.9  0.1  0.0;
         -0.1  0.8  0.2;
          0.0  0.0  0.7]
Q_true = 0.05 * Matrix(I(K_true))
b_true = zeros(K_true)

C_true = [1.0  0.5  0.0;
          0.8  0.3  0.1;
          0.2  1.0  0.0;
          0.0  0.7  0.4;
          0.1  0.2  0.9;
          0.3  0.0  0.8;
          0.6  0.4  0.2;
          0.4  0.6  0.5]
d_true = zeros(D)
R_true = 0.1 * Matrix(I(D))
x0_true = zeros(K_true)
P0_true = 0.2 * Matrix(I(K_true))

true_lds = LinearDynamicalSystem(;
    state_model=GaussianStateModel(A_true, Q_true, b_true, x0_true, P0_true),
    obs_model=GaussianObservationModel(C_true, R_true, d_true),
    latent_dim=K_true,
    obs_dim=D,
    fit_bool=fill(true, 6),
);

x_true, y_true = rand(rng, true_lds, T);

# ## Equivalent transformed copies
#
# A handful of similarity transforms: orthogonal rotations, an axis swap, a
# permutation, and a non-orthogonal scaling. All produce models with the
# same likelihood as the reference.

function rotate_lds(lds, R)
    A_rot = R * lds.state_model.A * inv(R)
    Q_rot = R * lds.state_model.Q * R'
    C_rot = lds.obs_model.C * inv(R)
    x0_rot = R * lds.state_model.x0
    P0_rot = R * lds.state_model.P0 * R'
    return LinearDynamicalSystem(;
        state_model=GaussianStateModel(A_rot, Q_rot, b_true, x0_rot, P0_rot),
        obs_model=GaussianObservationModel(C_rot, lds.obs_model.R, d_true),
        latent_dim=size(A_rot, 1),
        obs_dim=size(C_rot, 1),
        fit_bool=fill(true, 6),
    )
end

rotations = [
    [cos(π/4) -sin(π/4) 0.0;  sin(π/4) cos(π/4) 0.0;  0.0 0.0 1.0],
    [1.0 0.0 0.0; 0.0 cos(π/2) -sin(π/2); 0.0 sin(π/2) cos(π/2)],
    Matrix(qr(randn(rng, K_true, K_true)).Q),
    [0.0 0.0 1.0; 0.0 1.0 0.0; 1.0 0.0 0.0],
    Matrix(Diagonal([2.0, 0.5, -1.2])),
    [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0],
]
rot_names = [
    "rot(1,2, 45°)", "rot(2,3, 90°)", "random orthogonal",
    "axis swap (1↔3)", "scaling + sign", "permutation (1↔2)",
]

rotated_models = [rotate_lds(true_lds, R) for R in rotations]

# The invariant under any invertible $R$ is the *marginal* log-likelihood
# $\log p(y)$ (the latents are integrated out, so the volume Jacobian
# cancels). For a Gaussian LDS this equals the ELBO at the smoothed
# posterior — the same quantity [`fit!`](@ref) reports — so we assemble
# `calculate_elbo` directly. The function `loglikelihood(x_smooth, lds, y)`
# evaluates the *joint* `log p(x, y)` at the smoothed mean and is gauge-
# invariant only for orthogonal $R$; the diagonal scaling in `rotations`
# would shift it by $T \log |\det R|$.

function marginal_loglik(lds, y)
    tsteps_per_trial = [size(y, 2)]
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
    sws_pool = [
        StateSpaceDynamics.SmoothWorkspace(
            Float64, lds.latent_dim, lds.obs_dim, size(y, 2); ux_dim=0, uy_dim=0,
        ) for _ in 1:Threads.maxthreadid()
    ]
    suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
        Float64, lds, tsteps_per_trial,
    )
    ux_seq = [zeros(0, size(y, 2))]
    uy_seq = [zeros(0, size(y, 2))]
    StateSpaceDynamics._td_init_const_blocks!(
        sws_pool[1], lds, tsteps_per_trial, [y], ux_seq, uy_seq,
    )
    StateSpaceDynamics.smooth!(lds, tfs, [y], sws_pool, ux_seq, uy_seq)
    StateSpaceDynamics._aggregate_td_suff_stats!(
        suf, tfs, lds, ux_seq, uy_seq, [y], sws_pool[1],
    )
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)
    return StateSpaceDynamics.elbo!(lds, suf, sws_pool[1], total_entropy)
end

ll_orig = marginal_loglik(true_lds, y_true)

@printf("reference log p(y): %.6f\n", ll_orig)
for (name, R, model) in zip(rot_names, rotations, rotated_models)
    ll = marginal_loglik(model, y_true)
    @printf("%-22s  ΔLL = %.3e  cond(R) = %.2f\n",
        name, abs(ll - ll_orig), cond(R))
end

# ## Procrustes alignment
#
# To compare two equivalent fits visually we solve
# $\hat R = \arg\min_R \lVert R X_\text{rot} - X_\text{orig} \rVert_F$
# via SVD of $X_\text{orig} X_\text{rot}^\top$.

function procrustes_R(X, Y)
    S = svd(Y * X')
    return S.U * S.Vt
end

R_idx = 3
m_rot = rotated_models[R_idx]
x_orig, _ = smooth(true_lds, y_true)
x_rot, _ = smooth(m_rot, y_true)
Rhat = procrustes_R(x_rot, x_orig)

state_align_relerr = norm(Rhat * x_rot - x_orig) / norm(x_orig)
A_aligned = Rhat * m_rot.state_model.A * Rhat'
C_aligned = m_rot.obs_model.C * Rhat'
ΔA = norm(A_true - A_aligned)
ΔC = norm(C_true - C_aligned)

@printf("Procrustes residual: %.3e\n", state_align_relerr)
@printf("After alignment: ‖ΔA‖ = %.3e, ‖ΔC‖ = %.3e\n", ΔA, ΔC)

p_align = plot(layout=(2, 2), size=(900, 700))
heatmap!(p_align, A_true; title="A (true)", subplot=1, color=:RdBu, aspect_ratio=:equal)
heatmap!(p_align, A_aligned; title="A (aligned)", subplot=2, color=:RdBu, aspect_ratio=:equal)
heatmap!(p_align, C_true; title="C (true)", subplot=3, color=:RdBu, aspect_ratio=:equal)
heatmap!(p_align, C_aligned; title="C (aligned)", subplot=4, color=:RdBu, aspect_ratio=:equal)

# ## What *is* identifiable
#
# Even though individual coordinates are gauge-dependent, three things are
# invariant under similarity transforms: eigenvalues of $A$ (and therefore
# modal timescales), the column space of $C$ (up to subspace angles), and
# all predictive metrics.

function subspace_angles_deg(C1, C2)
    Q1 = qr(C1).Q[:, axes(C1, 2)]
    Q2 = qr(C2).Q[:, axes(C2, 2)]
    σ = clamp.(svdvals(Q1' * Q2), -1.0, 1.0)
    return acos.(σ) .* (180 / π)
end

λ_true = sort(abs.(eigvals(true_lds.state_model.A)))
for (name, model) in zip(rot_names, rotated_models)
    λ = sort(abs.(eigvals(model.state_model.A)))
    θ = subspace_angles_deg(C_true, model.obs_model.C)
    @printf("%-22s  max|Δ|λ|| = %.2e  max angle(C) = %.3f°\n",
        name, maximum(abs.(λ_true - λ)), maximum(θ))
end

# ## Tests  #src

using SSDTest  #src
using Test  #src

# All rotated models must have the same marginal log-likelihood.  #src
for model in rotated_models  #src
    @test isapprox(marginal_loglik(model, y_true), ll_orig; atol=1e-6)  #src
end  #src

# Procrustes residual must be essentially zero for orthogonal transforms.  #src
@test state_align_relerr < 1e-6  #src

# Eigenvalues of A are similarity-invariant.  #src
for model in rotated_models  #src
    λ = sort(abs.(eigvals(model.state_model.A)))  #src
    @test maximum(abs.(λ - λ_true)) < 1e-8  #src
end  #src
