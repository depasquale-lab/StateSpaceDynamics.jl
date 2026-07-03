# Benchmarking Guide

This document explains the benchmarking infrastructure for StateSpaceDynamics.jl.
For how to run things locally, see [`benchmark/README.md`](../benchmark/README.md).

## Overview

Benchmarking has two distinct jobs, which live side by side under `benchmark/`:

1. **Regression tracking** — [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
   runs `benchmark/benchmarks.jl` (the `SUITE`) on a PR and its base branch and
   reports any speed/allocation changes. This is the automated CI check.
2. **Cross-library comparison** — `benchmark/comparison/` benchmarks against the
   Python libraries pykalman and Dynamax. This is run manually (it needs a Conda
   Python environment), not in CI.

> AirspeedVelocity compares *this package across git revisions* — it cannot
> compare against other packages. The pykalman/Dynamax comparison is therefore a
> separate harness (`benchmark/comparison/`), not part of the AirspeedVelocity run.

## CI workflow

[`.github/workflows/airspeed.yml`](workflows/airspeed.yml) runs on every PR
(`pull_request_target`), benchmarks the PR head and its base in the same
environment via the `MilesCranmer/AirspeedVelocity.jl@action-v1` action, and posts
a comparison table as a PR comment. The action freezes the benchmark script at the
base revision, so PR code cannot rewrite what is measured.

## What gets benchmarked (regression suite)

`benchmark/benchmarks.jl` defines `const SUITE`:

- **Gaussian LDS smoothing** — latent ∈ {2, 4, 8}, obs ∈ {5, 10, 20}, T ∈ {100, 500}
- **Poisson LDS smoothing** — latent ∈ {2, 4}, obs ∈ {5, 10}, T ∈ {100, 500}

Metrics per benchmark: median time, allocated memory, allocation count.

## Adding a new benchmark

Edit `benchmark/benchmarks.jl` and add to `SUITE`:

```julia
SUITE["MyCategory"]["operation", "params"] =
    @benchmarkable my_function($args) samples = 10 seconds = 5
```

Use `StableRNG` for reproducibility, and cover small/medium/large problem sizes.

## Interpreting performance changes

| Verdict | Time | Memory |
|---------|------|--------|
| Good | speedup > 1.05 | reduction > 5% |
| Noise | 0.95–1.05 | < 5% |
| Investigate | speedup < 0.95 | increase > 10% |

## Troubleshooting

- **Times out** — reduce problem size, raise `seconds`, or lower `samples`.
- **Noisy** — raise `samples`; CI is usually more stable than a busy laptop.
- **OOM** — reduce problem sizes; CI runners have limited memory.
