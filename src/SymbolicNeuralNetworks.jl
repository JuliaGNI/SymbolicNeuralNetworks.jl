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

    include("equation_types.jl")

    export symbolize
    include("utils/symbolize.jl")

    include("utils/create_array.jl")

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

    export symbolic_hamiltonian
    include("hamiltonian.jl")

    export build_nn_function
    include("utils/build_function.jl")
    include("utils/build_function2.jl")
    include("utils/build_function_arrays.jl")

    export SymbolicPullback
    include("pullback.jl")

    include("derivatives/derivative.jl")
    include("derivatives/jacobian.jl")
    include("derivatives/gradient.jl")

    include("custom_equation.jl")

    include("utils/latexraw.jl")
end
