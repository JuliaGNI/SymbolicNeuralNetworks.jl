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

    include("symbolic_neuralnet/symbolize.jl")

    export AbstractSymbolicNeuralNetwork
    export SymbolicNeuralNetwork

    include("symbolic_neuralnet/symbolic_neuralnet.jl")

    export build_nn_function
    include("build_function/build_function.jl")
    include("build_function/build_function_double_input.jl")
    include("build_function/build_function_arrays.jl")

    export SymbolicPullback
    include("derivatives/pullback.jl")

    include("derivatives/derivative.jl")
    include("derivatives/jacobian.jl")
    include("derivatives/gradient.jl")

    include("custom_definitions_and_extensions/latexraw.jl")
end
