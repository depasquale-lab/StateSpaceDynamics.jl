# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
