using Documenter
using StateSpaceDynamics


DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics); recursive=true)

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
            "EmissionModels" => "EmissionModels.md",
            "Mixture Models" => "MixtureModels.md",
        ],
        "Miscellaneous" => "Misc.md",
    ]
)

deploydocs(; repo="github.com/depasquale-lab/StateSpaceDynamics.jl", devbranch="main")