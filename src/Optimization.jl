"""Optimization Utilities"""

export newton_raphson_tridg!

"""SGD-Step"""
#TODO: Add a SGD Step

"""ADAM Step"""
#TODO: Add an ADAM Step

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
    xₜ = Matrix{Float64}(reshape(xₜ', T*D, 1))
    # compute newton raphson step. The way this is coded we are given a vector that interleaves the dimensions of the state-space. E.g. for a 2x2 state-space,
    # where the state-space is given by [x₁, x₂], the vector would be [x₁₁, x₁₂, x₂₁, x₂₂]. This is done to match the Hessian and Gradient. The interleave_reshape fucntion undoes this interleaving.
    return interleave_reshape(xₜ - (Hessian \ Gradient), T, D)
end