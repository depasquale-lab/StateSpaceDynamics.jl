using Documenter
using Revise
using SSM

root = joinpath(dirname(pathof(SSM)), "..", "docs")
println("Current root: ", root)

entr([], [SSM]) do
    println("Current working directory: ", pwd())
    makedocs(root = root, sitename="SSM Julia")
end