"""
ACDC.jl

Accumulated Cutoff Discrepancy Criterion (ACDC) for robust model selection.

Based on Li et al. (2026), ACDC provides model selection that is robust to 
model misspecification by measuring component-level discrepancies via
the stochastic drivers framework.

Generative model:

```math
\\begin{aligned}
    x_n &= \\sum_k y_{n,k} \\\\
    y_{n,k} &= f(z_{n,k}, \\phi_k, \\epsilon_{n,k})\\qquad \\text{where} \\epsilon_{n,k} \\sim U(0,1)
\\end{aligned}
```

If the model is correct, recovered ε_{n,k} should be uniform.

The module is organized into:
- ACDCInterface.jl: Core types, stochastic_drivers interface, discrepancy measures
- ACDCAdapters.jl: Model-specific implementations for GMM, PMM, HMM, PPCA
"""
module ACDC

using LinearAlgebra
using Distributions
using Statistics
using NearestNeighbors
using SpecialFunctions
using ..StateSpaceDynamics

include("ACDCInterface.jl")
include("ACDCAdapters.jl")

end  # module
