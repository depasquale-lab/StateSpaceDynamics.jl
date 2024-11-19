using Documenter
using StateSpaceDynamics

root = joinpath(dirname(pathof(StateSpaceDynamics)), "..", "docs")

println("working dir: ", pwd())

DocMeta.setdocmeta!(
    StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics); recursive=true
)

makedocs(;
    root=root,
    sitename="StateSpaceDynamics Julia",
    modules=[StateSpaceDynamics],
    pages=[
        "Home" => "index.md",
        "getting_started.md",
        "Models" => [
            "BasicModels.md",
            "RegressionModels.md",
            "HiddenMarkovModels.md",
            "MixtureModels.md",
            "LDS.md",
        ],
        "Misc" => "Misc.md",
    ],
)
