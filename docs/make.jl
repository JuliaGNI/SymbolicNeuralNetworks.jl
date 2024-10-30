using SymbolicNeuralNetworks
using Documenter
using Latexify: LaTeXString

# taken from https://github.com/korsbo/Latexify.jl/blob/master/docs/make.jl
Base.show(io::IO, ::MIME"text/html", l::LaTeXString) = l.s

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
        mathengine = MathJax3()
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Vanilla Symbolic Neural Network" => "symbolic_neural_networks.md",
            "Hamiltonian Neural Network" => "hamiltonian_neural_network.md",
            ],
    ],
)

deploydocs(;
    repo   = "github.com/JuliaGNI/SymbolicNeuralNetworks.jl",
    devurl = "latest",
    devbranch = "main",
)
