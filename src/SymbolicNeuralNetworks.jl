module SymbolicNeuralNetworks

    using AbstractNeuralNetworks
    using Symbolics
    using LinearAlgebra

    import AbstractNeuralNetworks: NeuralNetwork, dim

    include("utils.jl")

    include("get_track.jl")

    export symbolic_params
    include("symbolic_params.jl")

    export AbstractSymbolicNeuralNetwork
    include("abstract_symbolic_neuralnet.jl")

    export SymbolicNeuralNetwork
    export architecture, params, models_params, neuralnet
    export Symbolize
    
    include("symbolic_neuralnet.jl")

    

    

end
