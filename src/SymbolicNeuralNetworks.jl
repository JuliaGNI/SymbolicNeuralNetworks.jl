module SymbolicNeuralNetworks

    using AbstractNeuralNetworks
    using Symbolics
    using LinearAlgebra

    import AbstractNeuralNetworks: NeuralNetwork

    include("symbolic_params.jl")

    include("abstract_symbolic_neuralnet.jl")

end
