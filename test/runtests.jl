using SymbolicNeuralNetworks
using SafeTestsets
using Test

# @safetestset "Symbolic Neural network                                                                " begin include("neural_network_derivative.jl") end
# @safetestset "Symbolic Params                                                                        " begin include("test_params.jl") end
# @safetestset "Develop/Envelop                                                                        " begin include("test_de_envelop.jl") end
# @safetestset "Get Track                                                                              " begin include("test_get_track.jl") end
# @safetestset "Rewrite code                                                                           " begin include("test_rewrite.jl") end
@safetestset "HNN Loss                                                                               " begin include("test_hnn_loss_pullback.jl") end
# @safetestset "Lagrangian                                                                             " begin include("test_lagrangian.jl") end
#= To add once GeometricMachineLearning is updated on the main branch
@safetestset "SymbolicNeuralNetwork                                                                  " begin include("test_symbolic_neuralnet.jl") end
@safetestset "Training SymbolicNeuralNetwork                                                         " begin include("test_training.jl") end
@safetestset "SymbolicNeuralNetwork with eqs                                                         " begin include("test_equations.jl") end
=#
