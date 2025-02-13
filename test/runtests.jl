using SymbolicNeuralNetworks
using SafeTestsets

@safetestset "Symbolic gradient                                                                      " begin include("derivatives/symbolic_gradient.jl") end
@safetestset "Symbolic Neural network                                                                " begin include("derivatives/jacobian.jl") end
@safetestset "Symbolic Params                                                                        " begin include("symbolic_neuralnet/symbolize.jl") end
@safetestset "Tests associated with 'build_function.jl'                                              " begin include("build_function/build_function.jl") end
@safetestset "Tests associated with 'build_function_double_input.jl'                                 " begin include("build_function/build_function_double_input.jl") end
@safetestset "Tests associated with 'build_function_array.jl                                         " begin include("build_function/build_function_arrays.jl") end
@safetestset "Compare Zygote Pullback with Symbolic Pullback                                         " begin include("derivatives/pullback.jl") end