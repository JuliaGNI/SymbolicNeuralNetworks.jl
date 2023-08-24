module SymbolicNeuralNetworks

    using AbstractNeuralNetworks
    using Symbolics
    using LinearAlgebra
    using RuntimeGeneratedFunctions
    using KernelAbstractions

    import AbstractNeuralNetworks: NeuralNetwork, Architecture, Model
    import AbstractNeuralNetworks: architecture, model, params
    

    include("utils.jl")

    export develop, envelop
    include("utils/de_envelop.jl")

    export get_track
    include("utils/get_track.jl")

    export rewrite_code
    include("utils/rewrite_code.jl")

    export symbolize
    include("utils/symbolize.jl")

    export symbolic_hamiltonian
    include("hamiltonian.jl")

    export symbolic_lagrangian
    include("lagrangian.jl")

    export AbstractSymbolicNeuralNetwork
    export SymbolicNeuralNetwork, SymbolicModel
    export architecture, model, params, equations, functions

    export symbolicparameters
    include("symbolicparameters.jl")

  
    include("symbolic_neuralnet.jl")

end
