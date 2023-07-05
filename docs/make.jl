using SymbolicNetworks
using Documenter

DocMeta.setdocmeta!(SymbolicNetworks, :DocTestSetup, :(using SymbolicNetworks); recursive=true)

makedocs(;
    modules=[SymbolicNetworks],
    authors="Michael Kraus",
    repo="https://github.com/JuliaGNI/SymbolicNetworks.jl/blob/{commit}{path}#{line}",
    sitename="SymbolicNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/SymbolicNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/SymbolicNetworks.jl",
    devbranch="main",
)
