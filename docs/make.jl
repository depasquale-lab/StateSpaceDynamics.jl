using Documenter
using StateSpaceDynamics
using Literate
using Random

# Set up the documentation environment
DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics, Random); recursive=true)

# Convert Gaussian LDS example
Literate.markdown(
    joinpath(@__DIR__, "examples", "GaussianLDS.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "gaussian_latent_dynamics_example",
    documenter = true
)

# Convert Poisson LDS example
Literate.markdown(
    joinpath(@__DIR__, "examples", "PoissonLDS.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "poisson_latent_dynamics_example",
    documenter = true
)

# Generate the documentation site
makedocs(;
    modules=[StateSpaceDynamics],
    authors="Ryan Senne",
    sitename="StateSpaceDynamics.jl",
    format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    repolink = "https://github.com/depasquale-lab/StateSpaceDynamics.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Linear Dynamical Systems" => "LinearDynamicalSystems.md",
            "Hidden Markov Models" => "HiddenMarkovModels.md",
            "Switching Linear Dynamical Systems" => "SLDS.md",
            "EmissionModels" => "EmissionModels.md",
            "Mixture Models" => "MixtureModels.md",
        ],
        "Tutorials" => [
            "Gaussian LDS Example" => "tutorials/gaussian_latent_dynamics_example.md",
            "Poisson LDS Example" => "tutorials/poisson_latent_dynamics_example.md"
        ],
        "Miscellaneous" => "Misc.md",
    ],
    checkdocs = :exports,
    warnonly = true
)

# Deploy the documentation
deploydocs(; repo="github.com/depasquale-lab/StateSpaceDynamics.jl", devbranch="docs_dev_")
