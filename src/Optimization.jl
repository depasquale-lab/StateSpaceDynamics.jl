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
    xₜ = Matrix{Float64}(reshape(xₜ', :, 1))
    # compute newton raphson step. The way this is coded we are given a vector that interleaves the dimensions of the state-space. E.g. for a 2x2 state-space,
    # where the state-space is given by [x₁, x₂], the vector would be [x₁₁, x₁₂, x₂₁, x₂₂]. This is done to match the Hessian and Gradient. The interleave_reshape fucntion undoes this interleaving.
    return interleave_reshape(xₜ - (Hessian \ Gradient), T, D)
end

"""
Newton-Raphson Routine assuming that the Hessian is block-tridiagonal. This is for an LDS model.

"""
function newton_raphson_tridg!(l::LDS, x0::AbstractArray, y::AbstractArray, niters::Int, tol::Float64=1e-6)
    # Initialize xₜ before the loop
    xₜ = copy(x0) 
    T, D = size(xₜ)
    # Run the Newton-Raphson routine
    for i in 1:niters
        # Compute the gradient
        grad = Gradient(l, y, xₜ)
        # Compute the Hessian
        hess = Hessian(l, y)
        # reshape the gradient to a vector to pass to newton_raphson_step_tridg!, we transpose as the way Julia reshapes is by vertically stacking columns as we need to match up observations to the Hessian.
        grad = Matrix{Float64}(reshape(grad', (T*D), 1))
        # Compute the Newton-Raphson step        
        xₜ₊₁ = newton_raphson_step_tridg!(xₜ, hess, grad)
        # Check for convergence (uncomment the following lines to enable convergence checking)
        if norm(xₜ₊₁ - xₜ) < tol
            println("Converged at iteration ", i)
            return xₜ₊₁
        else
            println("Norm of gradient iterate difference: ", norm(xₜ₊₁ - xₜ))
        end
        # Update the iterate
        xₜ = xₜ₊₁
    end
    # Print a warning if the routine did not converge
    println("Warning: Newton-Raphson routine did not converge.")
    return xₜ
end