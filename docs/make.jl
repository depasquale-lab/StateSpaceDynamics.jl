using Documenter
using StateSpaceDynamics
using Literate
using Random

# Set up the documentation environment
DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics, Random); recursive=true)

# Run Literate.jl to convert tutorial scripts to markdown in docs/src/tutorials/
Literate.markdown(
    joinpath(@__DIR__, "examples", "GaussianLDS.jl"), # input path
    joinpath(@__DIR__, "src", "tutorials");                        # output directory
    name = "latent_dynamics_example",                              # output .md filename
    documenter = true                                              # format for Documenter
)

# Generate the documentation site
makedocs(;
    modules=[StateSpaceDynamics],
    authors="Ryan Senne",
    sitename="StateSpaceDynamics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    remotes=nothing,
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
            "Gaussian LDS Example" => "tutorials/latent_dynamics_example.md"
        ],
        "Miscellaneous" => "Misc.md",
    ],
    checkdocs = :exports,
    warnonly = true
)

# Deploy the documentation
deploydocs(; repo="github.com/depasquale-lab/StateSpaceDynamics.jl", devbranch="docs_dev_")
