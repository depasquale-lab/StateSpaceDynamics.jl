# Benchmarking

All benchmarking lives under this directory. There are three distinct pieces:

| Path | Purpose | Deps |
|------|---------|------|
| `benchmarks.jl` + `Project.toml` | **Regression tracking** — the `SUITE` run by [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) to compare a PR against its base branch. | lightweight (BenchmarkTools, StableRNGs) |
| `comparison/` | **Cross-library comparison** — benchmarks StateSpaceDynamics against Python libraries (pykalman, Dynamax). | heavy (PythonCall/CondaPkg, Plots, CSV) |
| `profiling/` | **Ad-hoc profiling** scripts for chasing allocations / hot loops during development. | uses `comparison/`'s env |

## Regression tracking (AirspeedVelocity)

`benchmark/benchmarks.jl` defines `const SUITE::BenchmarkGroup` (Gaussian + Poisson
LDS smoothing across problem sizes). On every PR, the
[`.github/workflows/airspeed.yml`](../.github/workflows/airspeed.yml) workflow runs
the suite on both the PR head and its base in the same environment and posts a
comparison table as a PR comment.

Run it locally with the `benchpkg` CLI:

```bash
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'  # installs benchpkg into ~/.julia/bin
export PATH="$HOME/.julia/bin:$PATH"
benchpkg StateSpaceDynamics --rev=main,HEAD --bench-on=HEAD
```

To run the suite directly without comparison:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); using BenchmarkTools; run(SUITE; verbose=true)'
```

## Cross-library comparison (pykalman / Dynamax)

The `comparison/` harness builds equivalent models in StateSpaceDynamics, pykalman,
and Dynamax (the latter two via `CondaPkg.toml`) and benchmarks `fit!`/EM head to
head.

```bash
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path=".")'
julia --project=benchmark/comparison benchmark/comparison/run_benchmark.jl   # -> results/lds_benchmark_results.csv
julia --project=benchmark/comparison benchmark/comparison/plotting.jl        # -> results/*.svg/png
```

Curated comparison artifacts live in `comparison/results/`.

## Profiling

```bash
julia --project=benchmark/comparison benchmark/profiling/alloc_profile.jl
julia --project=benchmark/comparison benchmark/profiling/profile_allocations.jl
```
