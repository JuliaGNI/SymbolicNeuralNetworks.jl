module SymbolicNeuralNetworks

    using AbstractNeuralNetworks
    using Symbolics
    using LinearAlgebra
    using RuntimeGeneratedFunctions

    import AbstractNeuralNetworks: NeuralNetwork, dim

    export develop
    include("utils.jl")
    
    export get_track
    include("get_track.jl")

    export rewrite_code
    include("rewrite_code.jl")

    export symbolic_params
    include("params.jl")

    export symbolic_hamiltonian
    include("hamiltonian.jl")

    export symbolic_lagrangian
    include("lagrangian.jl")

    export AbstractSymbolicNeuralNetwork
    export SymbolicNeuralNetwork
    export Symbolize
    export architecture, params, models_params, neuralnet
    
    include("symbolic_neuralnet.jl")

    

    

    

end
