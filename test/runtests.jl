using SymbolicNeuralNetworks
using SafeTestsets
using Test

@safetestset "Docstring tests.                                                                      " begin include("doctest.jl") end
@safetestset "Symbolic gradient                                                                     " begin include("symbolic_gradient.jl") end
@safetestset "Symbolic Neural network                                                               " begin include("neural_network_derivative.jl") end
@safetestset "Symbolic Params                                                                       " begin include("test_params.jl") end
# @safetestset "HNN Loss                                                                               " begin include("test_hnn_loss_pullback.jl") end
@safetestset "Check if reshape works in the correct way with the generated functions.               " begin include("reshape_test.jl") end