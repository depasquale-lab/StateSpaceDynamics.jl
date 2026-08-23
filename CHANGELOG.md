# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `fit!(slds, y; tied_params=...)`: share any parameter group across every
  regime instead of fitting one per regime. Takes a `Symbol` or a collection of
  them, named the way `depends_on` and `fit_bool` name parameters — `[A b B]` is
  fit as one regression so any of `:A`/`:b`/`:B` names the whole group, likewise
  `:C`/`:d`/`:D` for `[C d D]`, with `:Q` and `:R` groups of their own.
  `tied_params = (:C, :R)` is the usual reading for neural data, where the
  recording does not change when the dynamics do (and `K` times fewer emission
  parameters); `tied_params = (:A, :Q)` is the mirror image, one set of dynamics
  with switching emissions. `:x0`/`:P0` are accepted and ignored, since an SLDS
  ties its initial state across regimes unconditionally. Works with
  `depends_on`, where the tie is *within* a group — each session keeps its own
  version, shared by every regime — and leaves a frozen group (`fit_bool`)
  untouched. Tied groups are broadcast before the first E-step, so no regime
  ever infers `q(x)`/`q(z)` through a parameter the model does not have
  * Tying a group alongside its noise covariance (`(:C, :R)`, `(:A, :Q)`) is the
    ordinary M-step on pooled statistics: the shared term does not depend on the
    regime and `Σₖ γₖ(t) = 1`, so the summed per-regime weighted objectives
    collapse to the unit-weight one. Tying only the noise is equally cheap —
    each regime contributes its own residual scatter and they are summed before
    the covariance is formed
  * Tying only the regression (`:C` without `:R`, `:A` without `:Q`) is exact
    but costs more. The residual covariance no longer divides out of `∂/∂W`, so
    the output rows couple and the shared fit becomes a generalized
    least-squares solve of size `p·m` — `O((p·m)³)` against the pooled fit's
    `O(m³)`. The solver (`_tied_gls_regression`) is written against a flat list
    of units and reduces exactly to the pooled `mn_map` when the covariances
    agree, so it is available to any future caller with the same shape
- `smooth(slds, y; ux, uy, depends_on, smoothing_iters, tol, return_cov,
  progress)`: the variational posteriors of a fitted SLDS at fixed parameters,
  returned as one `NamedTuple` `(; x, γ, elbo, p)` — the continuous states
  `q(x)`, the discrete responsibilities `γₜ(k) = q(zₜ = k)`, the ELBO at those
  posteriors, and (opt-in via `return_cov`) the smoothed covariances. It
  alternates forward-backward over the switching chain with the Laplace/Kalman
  smoother over the continuous states (Ghahramani & Hinton, 1996) until `γ`
  converges (`tol`) or `smoothing_iters` alternations are spent. Unlike the
  single-Monte-Carlo-sample E-step `fit!` runs during learning, the coupling
  here is deterministic — the discrete layer is scored at the smoothed
  posterior mean — so the result is reproducible with no `rng` to pass. `fit!`
  runs the same alternation but keeps its forward-backward storage private, so
  this is the way to get `q(z)` (regime occupancy, a Viterbi-style `argmax`
  path, a rate averaged over regimes) out of a model, on training or held-out
  data
- `fit!(slds, y; smoothing_iters=n)`: run `n` discrete↔continuous alternations
  per E-step instead of one. The default of 1 is the standard vLEM update;
  larger values hand the M-step a better-converged posterior at proportional
  cost per iteration
- Ancillary parameter dependencies: every
  `AbstractStateModel` and `AbstractObservationModel` now carries a
  `depends_on` field (default `nothing`). Setting it to a `NamedTuple` of
  per-trial label vectors — e.g. `obs_model.depends_on = (C = session, R =
  session)` — declares that those parameters are to be estimated separately for
  each group of trials, while everything else stays pooled. This is the
  "stitching" setup for combining recording sessions that observe different
  neurons in the same animal: shared latent dynamics, session-specific
  emissions
  * Keys are canonicalized to the same groups `fit_bool` uses, since those
    parameters are fit jointly as one regression — `:A`/`:b`/`:B` name one
    group and `:C`/`:d`/`:D` another. Labels may be `Symbol`s, integers or
    strings. Different parameters may use different label vectors; the trial
    partition is their common refinement
  * A malformed declaration (unknown parameter name, aliases of one group
    carrying different labels, label vectors of unequal length) is rejected by
    `validate_LDS` — so by the positional `LinearDynamicalSystem(state_model,
    obs_model)` constructor, not at the first `fit!`
  * Per-group values live in a new `variants` field holding one model object
    per parameter-group combination, with non-varying parameters shared **by
    reference** so a single M-step write covers all of them. They are read back
    with the new exported `group_labels(model, name)` and
    `group_parameter(model, name, label)`
  * `show` reports the declared groups for a model that has any
  * Supported for the **Gaussian LDS**, the **Poisson LDS** and the **SLDS**
    across `fit!`, `smooth`, `elbo`, `loglikelihood` and `rand`, each of which
    also accepts a `depends_on` keyword overriding the model's stored labels so
    a held-out set with a different trial count can be scored without mutating
    the model. All versions of a parameter share the model's prior; each
    version contributes its own log-prior term to the ELBO
  * The efficiency of same-length epochs is preserved *within* each group:
    trials sharing every parameter form a cell, and the smoothed covariance is
    computed once per cell and shared across it (parameters differ between
    cells, so their covariances genuinely differ). The `O(D²·T)` workspace
    storage is allocated once and reused across cells, so a grouped fit's
    memory tracks an ungrouped one's instead of scaling with the number of
    groups
  * For the Poisson emission there is no sufficient-statistic form, so the
    M-step runs one LBFGS solve per version of `[C d D]`, over the trials of
    every cell that shares that version
  * For an `SLDS` every regime must declare the same labels — the trial
    partition is a property of the data, not of a regime — and mismatched
    declarations raise an `ArgumentError`; `x0`/`P0` stay tied across regimes
    as they are for an ungrouped SLDS
  * With `depends_on` unset, every entry point takes its original code path
- Public allocating `elbo(model, y; ...)` for all three models (Gaussian LDS
  with `ux`/`uy` keywords, Poisson LDS with Newton-smoother keywords, SLDS
  with an `rng` keyword since its E-step consumes a posterior sample). Runs
  one E-step and evaluates the ELBO at the resulting posterior, the same
  quantity `fit!` reports per iteration,  without requiring the private
  workspace structs the `elbo!` variants take (#139)
- `loglikelihood(slds, y)` now throws an informative error (the marginal is
  intractable for a switching model; use `elbo`) instead of a raw
  `MethodError`, mirroring the Poisson LDS (#139)
- Composable Normal-Inverse-Wishart prior on the initial latent state via a new
  `GaussianStateModel` field `x0_prior::Union{Nothing,MNPrior}` (the mean half),
  paired with the existing `P0_prior::IWPrior` (the covariance half). The initial
  state `x₁ ~ N(x0, P0)` is an intercept-only regression, so its NIW prior is the
  same `MNPrior` + `IWPrior` composition used for `[A b]`/`Q` and `[C d]`/`R` —
  no bespoke prior type. Construct the mean half with the exported
  `x0_mean_prior(μ₀; κ₀)` helper; the M-step then does
  `x0 = (Σγ·x₁ + κ₀ μ₀) / (Σγ + κ₀)` and folds `κ₀(x0-μ₀)(x0-μ₀)'` into the IW
  scale. With `κ₀ → 0` (and no `P0_prior`) it reduces exactly to the previous MLE
  update

### Changed
- **Breaking:** the previously exported (but unused) `Data` struct is now a
  private, validated container for multi-trial observations + `ux`/`uy` inputs.
  Public entry points (`fit!`, `smooth`, `loglikelihood`) accept plain arrays —
  a `(obs_dim, T)` matrix, a `(obs_dim, T, ntrials)` array, or a vector of
  per-trial matrices, with `ux`/`uy` in the same shape family — and construct
  a `Data` at the boundary, which is the single shape/dimension validation
  site (observation rows are now checked against `obs_dim` up front). The
  multi-trial backend (`estep!`, multi-trial `smooth!`, the sufficient-stats
  aggregators, `_fit_tridiag!`) consumes `Data` instead of threading
  `y`/`ux`/`uy` triples through every signature (#139)
- `fit!(slds, y)` now validates observations through `Data` like the other
  entry points (dimension mismatches throw a clean `DimensionMismatchError`
  upfront instead of failing deep in the smoother) and accepts the
  `(obs_dim, T, ntrials)` array form (#139)
- `smooth` (public, allocating) now accepts `ux`/`uy` keywords on the Gaussian
  path and all three observation shapes on both the Gaussian and Poisson
  paths; multi-trial input returns per-trial vectors, matrix input returns
  matrices as before (#139)
- **Breaking:** `elbo(slds, y)` is now deterministic. It infers `q(x)` and
  `q(z)` by the same coordinate ascent as `smooth(slds, y)` and returns that
  call's `elbo` field, rather than running one Monte-Carlo E-step off a joint
  draw from `q(x)`. It no longer takes an `rng`, and takes `smoothing_iters` /
  `tol` / `progress` instead; the value is a converged bound rather than one
  matching `fit!`'s first noisy trace entry
- **Breaking:** `loglikelihood(slds, y)` returns the ELBO instead of throwing.
  The exact marginal `log p(y)` is still intractable for a switching model
  (it needs a sum over all `K^T` regime sequences), so the returned value is a
  variational lower bound — comparable across models fit to the same data, but
  not a likelihood
- **Breaking:** renamed the control-input arguments `latent_inputs`/`obs_inputs`
  to `ux`/`uy` across the public API (keywords on `fit!`/`rand`, positional on
  `smooth!`/`estep!`) (#139)
- **Breaking:** renamed the `LinearDynamicalSystem` fields
  `state_input_dim`/`obs_input_dim` to `ux_dim`/`uy_dim` to match (#139)
- Multithreading now uses OhMyThreads.jl (`tforeach`/`tmapreduce`) instead of
  `Base.Threads` (`@threads`/`@spawn`); OhMyThreads is a new dependency (#143)
- Deduplicated the per-observation-model complete-data log-likelihood, gradient,
  and Hessian implementations (Gaussian / Poisson / SLDS-weighted) into shared
  kernels in `continuous_latents.jl`. The emission-specific pieces are now single
  dispatch points (`obsloglikelihood!`, `observationgradient!`, and the Hessian
  emission block), the affine transition residual is defined in exactly one
  place, and kernel signatures follow a uniform `f!(out, ws, model, x, y, ...)`
  convention — a new observation model plugs in with one method per kernel
  (#135, #136, #141)
- Reworked the monolithic ~90-field `SmoothedWorkspace` into modular components
  (`BlockTridiagonalWorkspace`, `SmoothConstants`, `NewtonBuffers`,
  `RegressionBuffers`, `ElboBuffers`, `TDAggBuffers`, `BatchedBuffers`) and
  unified the dev-facing function handles across the Gaussian, Poisson, and
  SLDS paths (#144)
- `loglikelihood(lds, y)` now computes the observation-independent half of the
  Kalman filter (innovation covariances and gains) once and shares it across
  trials, uses the positive-definite-by-construction information-form update
  (`info_update!`) from the retired Kalman path, supports ragged trial
  lengths, and accepts `ux`/`uy` input keywords. Models with input matrices
  (`B`/`D` with nonzero columns) now **require** the matching input
  sequences — previously inputs were silently ignored, giving a wrong
  likelihood

### Removed
- Stale one-off profiling scripts under `benchmark/profiling/` (#144)
- The retired information-form Kalman/RTS smoother EM machinery
  (`src/stats/kalman.jl`: the `_fit_kalman!` driver with its E/M-step, ELBO,
  and sufficient-statistics code, plus the internal `KalmanWorkspace`). It had
  not been a selectable `fit!` backend since v0.4.0; the filter it contributed
  now lives behind `loglikelihood` (see Changed). `marginal_loglikelihood`
  remains as an internal alias of `loglikelihood`
- The internal `tol_PD` / `id_PD` eigen-floor helpers. The filter wraps the
  model covariances as strict `PDMat`s instead. — a genuinely non-PD `Q`/`R`/
  `P0` now fails.

### Fixed
- A grouped (`depends_on`) fit that pooled a regression over units with
  *different* noise versions — e.g. `depends_on = (R = session,)` with one
  emission over all sessions — solved the ordinary pooled normal equations,
  which do not maximize the ELBO when the residual covariance is not shared.
  Those cases now go through the generalized-least-squares solve described
  under Added; a version whose units do share a covariance keeps the cheap
  pooled path, which is the same estimator
- The documentation build failed: `set_group_seeds!` is exported and carries a
  docstring but was not in any `@docs` block, so Documenter raised both
  `missing_docs` and the unresolved `@ref`s pointing at it
- Multi-trial `rand(lds, tsteps_per_trial)` threw
  `Attempted to capture and modify outer local variables` instead of sampling.
  The per-trial parameter vectors were assigned from two branches of an `if`
  and then captured by the `tforeach` sampling closure, so Julia boxed them and
  OhMyThreads rejected the closure outright. They are now built in a helper, so
  each name is assigned once
- `fit!(slds, y; tie_emissions=true)` fitted the shared Gaussian emission from an
  uninitialized workspace, usually throwing `PosDefException` from the Cholesky
  in `_aggregate_td_suff_stats!` and otherwise returning nonsense. The
  unit-weight aggregator seeds its buffers from the data-only constant blocks
  (`Σ y y'`, `Σ y`, the observation count, the `uy` blocks), which only the
  LDS/PLDS `fit!` entry points fill — the SLDS never reaches them, because its
  own M-step goes through the weighted aggregator, which needs no constants.
  Both the plain and the grouped tied-emission updates now fill them first
- SLDS `forward_backward` could produce `NaN`s when a regime received ~no
  responsibility at trial starts: its initial-state effective count `init_n`
  underflowed toward zero, so `x0 = init_xy/init_n` and `P0 = S0/init_n` blew up
  to `±Inf`/`NaN`, poisoning `dl.logL` and the whole chain posterior. Setting the
  new initial-state NIW prior (`x0_prior` + `P0_prior`) makes the update degrade
  to the prior (`x0 → μ₀`, `P0 →` its IW mode) instead of dividing by ~0
- SLDS ELBO computation (incorrect sign, among other errors); correctness is now
  tested via the K=1 SLDS ≡ LDS equivalence (#145)
- SLDS posterior sampling drew from the marginals of `q(x)` (a mean-field
  approximation) instead of the joint smoothed posterior (#145)
- SLDS complete-data log-likelihood was inconsistent with the LDS
  implementations; fixed by deduplicating into the shared kernels, with new
  tests against Distributions.jl (#135)
- Poisson LDS ELBO omitted the matrix-normal prior term on the stacked dynamics
  `[A b B]` (#146)
- PPCA M-step computed the `σ²` update from the stale `W` instead of the freshly
  updated one (#147)
- The backtracking line search's cubic interpolation always stepped to the local
  minimizer of the interpolant, even when maximizing (#147)
- `validate_probvec` used a tolerance that could give wrong results for
  lower-precision element types (e.g. `Float32`) (#147)
- `tol_PD` threw a method error when the `tol` keyword did not match the matrix
  element type (#147)
- The LDS model-selection docs example could select the wrong latent
  dimension (#148)

## [0.4.1] - 2026-07-07

### Added
- Add `tview` helper to fix JET errors
- LineSearches added as a new dep, since `Optim` no longer re-exposes `HagerZhang` (#89)

### Changed
- Tests and code now use `Optim` v2 API (#89)
- Updates `Optim` lower bound `2` (#89)
- `Optim` v2's `LBFGS`+`HagerZhang` line search converges to marginally different values for the Poisson observation M-step (~1e-5 magnitude) than v1 did (#89)
- `Symmetrize!` now returns a `Symmetric` matrix (#130)

### Fixed
- Re-enables JET testing after fixing false positives. (#124)
- Fixes lower bound of Julia to 1.10 (was 1.11) (#89)
- Enforced a stable A matrix in the SLDS tests to fix flakyness in CI testing. (#130)
- Enforces symmetry in certain SLDS statistics causing a non-PD issue. (#130)

## [0.4.0] - 2026-07-03

### Added

- CHANGELOG.md to track version history
- Benchmarking CI workflow to track performance over time
- Centralized exports in StateSpaceDynamics.jl main module
- Custom exception types with improved error messages:
  - `DimensionMismatchError` for dimension validation
  - `NotPositiveDefiniteError` for matrix validation
  - `NotSymmetricError` for symmetry checks
  - `InvalidProbabilityVectorError` for probability vector validation
  - `NumericalStabilityError` for numerical issues
- Matrix-normal priors (`MNPrior`) on the stacked dynamics `[A b B]` and
  emission `[C d D]` matrices, giving a full MNIW MAP when paired with `IWPrior`
- Support for exogenous inputs: a dynamics input matrix `B` (`B·u`) and an
  observation input matrix `D` (`D·v`), with explicit `b` / `d` bias vectors
- Hand-rolled Newton smoother (`newton_smooth!`) with a backtracking line
  search for the non-conjugate (Poisson) observation path
- QuickStart example/tutorial
- Auto-formatting CI workflows (`Format.yml`, `Format-PR.yml`)

### Changed

- Refactored model validation system with descriptive exceptions
- Improved error messages across validation functions
- Consolidated all package exports into main module file
- Refactored block tridiagonal inverse implementation
- Renamed the `PoissonObservationModel` field `log_d` to `d`, adopting the
  canonical log-link `λ = exp(C x + d)`
- Standardized `fit_bool` layout: length 6 for the Gaussian path
  (`[x0, P0, A&b&B, Q, C&d&D, R]`) and length 5 for the Poisson path
  (`[x0, P0, A&b, Q, C&d]`)
- Reorganized the LDS source tree, extracting shared emission-agnostic code out
  of `gaussian.jl` into `common.jl` (parameter extraction / FilterSmooth init),
  `simulate.jl` (sampling), `dynamics.jl` (state M-step and state ELBO term), and
  `suff_stats.jl` (sufficient-statistics aggregation); moved the block-tridiagonal
  kernel into `block_tridiagonal.jl`, control-input validation into the validation
  module, and `Base.show` methods into `show.jl`
- Substantially optimized the multi-trial EM hot path: sufficient-statistics
  aggregation that is O(1) in trial length `T` and trial count `N`, a shared
  smoothed-covariance cache for equal-length trials, and an allocation-minimal
  block-tridiagonal smoother
- Clarified the log-likelihood API: the complete-data `log p(x, y)` given a
  trajectory is now `joint_loglikelihood(x, lds, y)`, while `loglikelihood(lds, y)`
  is the marginal (observed-data) `log p(y); a method of `StatsAPI.loglikelihood`,
  consistent with `loglikelihood(ppca, X)`. The marginal throws for Poisson LDS
  (intractable). Replaces the former `filter_loglikelihood`.

### Removed

- **Refocused the package on Linear Dynamical Systems.** Removed the Hidden
  Markov Model, Mixture Model, and standalone emission/regression model families
  along with their tests, documentation, examples, and benchmarks. Specifically:
  - Hidden Markov Models and GLM-HMMs: `HiddenMarkovModel`, `viterbi`,
    `class_probabilities`, the switching Gaussian/Poisson/Bernoulli regression
    models, and AutoRegressive HMM (ARHMM) support
  - Mixture Models: `GaussianMixtureModel`, `PoissonMixtureModel`
  - Emission / regression models: `EmissionModel`, `GaussianEmission`,
    `RegressionEmission`, `GaussianRegressionEmission`,
    `BernoulliRegressionEmission`, `PoissonRegressionEmission`,
    `AutoRegressionEmission`
- The Kalman/RTS smoother as a selectable E-step backend for `fit!` (the
  `kalman_filter` flag on `LinearDynamicalSystem`). All Gaussian fitting now uses
  the block-tridiagonal MAP path. The Kalman filter implementation is retained
  internally for the marginal log-likelihood `loglikelihood(lds, y)`.

### Fixed

- Formatter issues in test suite
- Documentation consistency across modules
- Double-exponential bug in the Poisson observation rate (previously
  `exp(C x + exp(log_d))`, now `exp(C x + d)`)

## [0.3.0] - 2025-11-12

### Added

- Inverse-Wishart priors for covariance matrices (IWPrior)
- Support for MAP estimation with priors on Q, P0, and R matrices
- PoissonLDS prior functionality
- JET.jl static analysis integration in CI
- Comprehensive test suite for prior-based estimation

### Changed

- Refactored LDS code structure for better maintainability
- Split LDS implementations into separate files (gaussian.jl, poisson.jl, types.jl)
- Improved test organization with shared utilities

### Fixed

- Block tridiagonal inverse numerical stability
- Test runner organization

## [0.2.0] - 2024-06-18

### Added

- Documentation improvements
- Enhanced plotting capabilities in examples
- DOI badge and updated README

### Changed

- Updated documentation structure
- Improved badges and metadata

## [0.1.0] - 2024-04-10

### Added

- Initial release of StateSpaceDynamics.jl
- Core implementations:
  - Linear Dynamical Systems (Gaussian and Poisson observations)
  - Hidden Markov Models (Gaussian, Poisson, ARHMM)
  - Mixture Models (Gaussian, Poisson)
  - Switching Linear Dynamical Systems (SLDS)
  - HMM-GLMs (Gaussian, Poisson, Bernoulli)
- Inference algorithms:
  - Kalman filtering and RTS smoothing
  - Laplace approximation for non-conjugate models
  - EM algorithm for parameter estimation
  - Forward-backward algorithm for HMMs
  - Viterbi algorithm for state sequences
- Utilities:
  - K-means initialization
  - Block tridiagonal matrix operations
  - Covariance matrix stabilization
  - Probabilistic PCA preprocessing
- Validation framework
- Comprehensive test suite
- Documentation and examples
- Benchmarking suite

[Unreleased]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/releases/tag/v0.1.0
