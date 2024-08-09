using Documenter
using SSM

root = joinpath(dirname(pathof(SSM)), "..", "docs")

println("working dir: ", pwd())

DocMeta.setdocmeta!(SSM, :DocTestSetup, :(using SSM); recursive=true)


makedocs(
    root = root, 
    sitename="SSM Julia",
    modules = [SSM],
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
        ])
