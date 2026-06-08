# Pretty-printing (`Base.show`) for the LDS / SLDS model types. Extracted from
# types.jl so the type definitions and constructors stay free of display logic.
# Included after types.jl because each method signature references a type defined
# there.

# Pretty print function that doesn't truncate arrays of model objects

"""
    print_full([io::Union{IO, Base.TTY}, ] obj)

Prints full description of object `obj`, overriding both `io`-based limits as
well as the limits set in the default pretty printing of `StateSpaceDynamics`
objects.
"""
function print_full(io::Union{IO,Base.TTY}, obj)
    println(IOContext(io, :limit => false), obj)

    return nothing
end

print_full(obj) = print_full(stdout, obj)

function Base.show(io::IO, gsm::GaussianStateModel; gap="")
    println(io, gap, "Gaussian State Model:")
    println(io, gap, "---------------------")

    if size(gsm.A, 1) > 4 || size(gsm.A, 2) > 4
        println(io, gap, " State Parameters:")
        println(io, gap, "  size(A)  = ($(size(gsm.A,1)), $(size(gsm.A,2)))")
        println(io, gap, "  size(Q)  = ($(size(gsm.Q,1)), $(size(gsm.Q,2)))")
        println(io, gap, " Initial State:")
        println(io, gap, "  size(b)  = ($(length(gsm.b)), )")
        println(io, gap, "  size(x0) = ($(length(gsm.x0)), )")
        println(io, gap, "  size(P0) = ($(size(gsm.P0,1)), $(size(gsm.P0,2)))")
    else
        println(io, gap, " State Parameters:")
        println(io, gap, "  A  = $(round.(gsm.A, sigdigits=3))")
        println(io, gap, "  Q  = $(round.(gsm.Q, sigdigits=3))")
        println(io, gap, " Initial State:")
        println(io, gap, "  b  = $(round.(gsm.b, digits=2))")
        println(io, gap, "  x0 = $(round.(gsm.x0, digits=2))")
        println(io, gap, "  P0 = $(round.(gsm.P0, sigdigits=3))")
    end

    println(io, gap, " Dynamics input:")
    println(io, gap, "  size(B)  = ($(size(gsm.B,1)), $(size(gsm.B,2)))")

    return nothing
end

function Base.show(io::IO, gom::GaussianObservationModel; gap="")
    println(io, gap, "Gaussian Observation Model:")
    println(io, gap, "---------------------------")

    if size(gom.C, 1) > 3 || size(gom.C, 2) > 3
        println(io, gap, " size(C) = ($(size(gom.C,1)), $(size(gom.C,2)))")
        println(io, gap, " size(R) = ($(size(gom.R,1)), $(size(gom.R,2)))")
        println(io, gap, " size(d) = ($(length(gom.d)),)")
        println(io, gap, " size(D) = ($(size(gom.D,1)), $(size(gom.D,2)))")
    else
        println(io, gap, " C = $(round.(gom.C, digits=2))")
        println(io, gap, " R = $(round.(gom.R, digits=2))")
        println(io, gap, " d = $(round.(gom.d, digits=2))")
        println(io, gap, " D = $(round.(gom.D, digits=2))")
    end

    return nothing
end

function Base.show(io::IO, pom::PoissonObservationModel; gap="")
    nobs, nstate = size(pom.C)

    println(io, gap, "Poisson Observation Model:")
    println(io, gap, "--------------------------")

    if nobs > 4 || nstate > 4
        println(io, gap, " size(C) = ($nobs, $nstate)")
        println(io, gap, " size(d) = ($(length(pom.d)),)")
    else
        println(io, gap, " C    = $(round.(pom.C, digits=2))")
        println(io, gap, " d    = $(round.(pom.d, sigdigits = 3))")
        println(
            io,
            gap,
            " rate = $(round.(exp.(pom.d), digits = 2))   # exp(d) for inspection only",
        )
    end

    return nothing
end

function Base.show(io::IO, lds::LinearDynamicalSystem; gap="")
    println(io, gap, "Linear Dynamical System:")
    println(io, gap, "------------------------")
    Base.show(io, lds.state_model; gap=gap * " ")
    Base.show(io, lds.obs_model; gap=gap * " ")
    println(io, gap, " Parameters to update:")
    println(io, gap, " ---------------------")

    if lds.obs_model isa PoissonObservationModel
        # C and d are either both updated or neither
        prms = ["x0", "P0", "A (and b)", "Q", "C, d"][lds.fit_bool[1:5]]
    else
        # Same labels for BTD and Kalman backends (length 6). The compound
        # entries "A (and b, B)" / "C (and d, D)" reflect that each row is
        # fit jointly as one regression — the bias and user-input columns
        # are not gated independently.
        prms = ["x0", "P0", "A (and b, B)", "Q", "C (and d, D)", "R"][lds.fit_bool[1:6]]
    end

    println(io, gap, "  $(join(prms, ", "))")
    return nothing
end

function Base.show(io::IO, slds::SLDS; gap="")
    K = length(slds.LDSs)

    println(io, gap, "Switching Linear Dynamical System (SLDS):")
    println(io, gap, "-----------------------------------------")
    println(io, gap, " Number of discrete states: $K")

    if K > 3
        println(io, gap, " size(A)  = ($(size(slds.A,1)), $(size(slds.A,2)))")
        println(io, gap, " size(πₖ) = ($(length(slds.πₖ)),)")
    else
        println(io, gap, " A  = $(round.(slds.A, sigdigits=3))")
        println(io, gap, " πₖ = $(round.(slds.πₖ, sigdigits=3))")
    end

    println(io, gap, " Linear Dynamical Systems:")
    println(io, gap, " -------------------------")

    # Show details of first LDS
    if K > 0
        println(io, gap, "  State 1:")
        Base.show(io, slds.LDSs[1]; gap=gap * "   ")

        if K > 1
            println(io, gap, "  ... and $(K-1) more state(s)")
        end
    end

    return nothing
end
