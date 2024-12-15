using SymbolicNeuralNetworks
using SafeTestsets
using Test

@safetestset "Docstring tests.                                                                       " begin include("doctest.jl") end
@safetestset "Symbolic gradient                                                                      " begin include("derivatives/symbolic_gradient.jl") end
@safetestset "Symbolic Neural network                                                                " begin include("derivatives/jacobian.jl") end
@safetestset "Symbolic Params                                                                        " begin include("symbolic_neuralnet/symbolize.jl") end
@safetestset "Tests associated to 'build_function.jl'                                                " begin include("build_function/build_function.jl") end