using Documenter
using Revise
using SSM

root = joinpath(dirname(pathof(SSM)), "..", "docs")

println("working dir: ", pwd())

entr(["./docs/src"], [SSM]) do
    makedocs(
        root = root, 
        sitename="SSM Julia",
        pages=[
            "Home" => "index.md",
            "getting_started.md",
            "Models" => [
                "MixtureModels.md",
                "RegressionModels.md",
                "HiddenMarkovModels.md",
                "LDS.md",
                ],
            ])
end