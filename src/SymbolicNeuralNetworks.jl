module SymbolicNeuralNetworks

    using AbstractNeuralNetworks
    using Symbolics
    using LinearAlgebra
    using RuntimeGeneratedFunctions
    using KernelAbstractions
    using AbstractNeuralNetworks: QPTOAT

    import AbstractNeuralNetworks: NeuralNetwork, Architecture, Model, UnknownArchitecture, AbstractExplicitLayer, NeuralNetworkParameters
    import AbstractNeuralNetworks: architecture, model, params
    # these types will be shifted to `GeometricOptimizers` once this package is ready
    import AbstractNeuralNetworks: NetworkLoss, AbstractPullback
    import Symbolics: NaNMath
    import Latexify: latexify
    import Zygote
    import ChainRulesCore
    
    RuntimeGeneratedFunctions.init(@__MODULE__)

    export optimize_code!
    include("utils/optimize_code.jl")

    export develop, envelop
    include("utils/de_envelop.jl")

    export get_track
    include("utils/get_track.jl")

    export rewrite_code
    include("utils/rewrite_code.jl")

    export symbolize
    include("utils/symbolize.jl")

    export AbstractSymbolicNeuralNetwork
    export SymbolicNeuralNetwork, SymbolicModel
    export HamiltonianSymbolicNeuralNetwork, HNNLoss
    export architecture, model, params, equations, functions

    # make symbolic parameters (`NeuralNetworkParameters`)
    export symbolicparameters
    include("layers/abstract.jl")
    include("layers/dense.jl")
    include("layers/linear.jl")
    include("chain.jl")

    export evaluate_equations
    include("symbolic_neuralnet.jl")

    include("neuralnet.jl")

    export symbolic_hamiltonian
    include("hamiltonian.jl")

    export symbolic_lagrangian
    include("lagrangian.jl")

    export build_nn_function
    include("utils/build_function.jl")

    export SymbolicPullback
    include("pullback.jl")

    export parallelize_expression, parallelize_expression_inplace, parallelize_pullback!
    include("parallelize_expression.jl")

    include("derivatives/derivative.jl")
    include("derivatives/jacobian.jl")
    include("derivatives/gradient.jl")
end
