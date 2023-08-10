using SymbolicNeuralNetworks
using Documenter

DocMeta.setdocmeta!(SymbolicNeuralNetworks, :DocTestSetup, :(using SymbolicNeuralNetworks); recursive=true)

makedocs(;
    modules=[SymbolicNeuralNetworks],
    authors="Michael Kraus",
    repo="https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/blob/{commit}{path}#{line}",
    sitename="SymbolicNeuralNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/SymbolicNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo   = "github.com/JuliaGNI/SymbolicNeuralNetworks.jl",
    devurl = "latest",
    devbranch = "main",
)
