module SymbolicNeuralNetworks

    using AbstractNeuralNetworks
    using Symbolics
    using LinearAlgebra
    using RuntimeGeneratedFunctions
    using AbstractNeuralNetworks: QPTOAT

    import Latexify: _latexraw
    import AbstractNeuralNetworks: NeuralNetwork, Architecture, Model, UnknownArchitecture, AbstractExplicitLayer, NeuralNetworkParameters
    import AbstractNeuralNetworks: architecture, model, params
    # these types will be shifted to `GeometricOptimizers` once this package is ready
    import AbstractNeuralNetworks: NetworkLoss, AbstractPullback
    import Symbolics: NaNMath
    
    RuntimeGeneratedFunctions.init(@__MODULE__)

    include("custom_definitions_and_extensions/equation_types.jl")

    export symbolize
    include("symbolic_neuralnet/symbolize.jl")

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
    include("symbolic_neuralnet/symbolic_neuralnet.jl")

    export build_nn_function
    include("build_function/build_function.jl")
    include("build_function/build_function2.jl")
    include("build_function/build_function_arrays.jl")

    export SymbolicPullback
    include("derivatives/pullback.jl")

    include("derivatives/derivative.jl")
    include("derivatives/jacobian.jl")
    include("derivatives/gradient.jl")

    include("custom_definitions_and_extensions/latexraw.jl")
end
