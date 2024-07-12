"""Optimization Utilities"""

export newton_raphson_tridg!

"""SGD-Step"""
#TODO: Add a SGD Step

"""ADAM Step"""
#TODO: Add an ADAM Step

"""
    line_search(x, Δx, f, α=1.0, β=0.5, σ=1e-4)

Perform a line search to find an appropriate step size for optimization.

Arguments:
- `x`: The current point in the optimization process.
- `Δx`: The search direction.
- `f`: The objective function to be minimized.
- `α`: The initial step size. Default is 1.0.
- `β`: The step size reduction factor. Default is 0.5.
- `σ`: The sufficient decrease parameter. Default is 1e-4.

Returns:
- `α`: The step size that satisfies the sufficient decrease condition.

The line search algorithm iteratively reduces the step size `α` until the sufficient decrease condition is satisfied. The sufficient decrease condition is defined as `f(x + α * Δx) <= f(x) + σ * α * Δϕ`, where `Δϕ` is the directional derivative of `f` at `x` along the direction `Δx`.

This function is commonly used in optimization algorithms to find an appropriate step size for updating the current point in the optimization process.
"""
function line_search(x, Δx, f, α=1.0, β=0.5, σ=1e-4)
    ϕ = f(x)
    Δϕ = dot(Δx, Δx)
    while f(x + α * Δx) > ϕ + σ * α * Δϕ
        α *= β
    end
    return α
end

"""
Newton-Raphson Step assuming that the Hessian is block-tridiagonal.

Arguments:
- `xₜ::AbstractArray`: Current iterate
- `Hessian::AbstractArray`: Hessian matrix
- `Gradient::AbstractArray`: Gradient vector

Returns:
- `xₜ₊₁::AbstractArray`: Next iterate
"""
function newton_raphson_step_tridg!(xₜ::AbstractArray, Hessian::AbstractArray, Gradient::AbstractArray)
    T, D = size(xₜ)
    # reshape xₜ so we can subtract
    xₜ = Matrix{Float64}(reshape(xₜ', :, 1))
    # compute newton raphson step. The way this is coded we are given a vector that interleaves the dimensions of the state-space. E.g. for a 2x2 state-space,
    # where the state-space is given by [x₁, x₂], the vector would be [x₁₁, x₁₂, x₂₁, x₂₂]. This is done to match the Hessian and Gradient. The interleave_reshape fucntion undoes this interleaving.
    return interleave_reshape(xₜ - (Hessian \ Gradient), T, D)
end