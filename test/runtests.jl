using SymbolicNeuralNetworks
using SafeTestsets
using Test

@testset "SymbolicNeuralNetworks.jl" begin
   
    #@safetestset "Symbolic Params                                                                        " begin include("test_params.jl") end
    @safetestset "Get Track                                                                              " begin include("test_get_track.jl") end
    @safetestset "Rewrite code                                                                           " begin include("test_rewrite.jl") end
    @safetestset "Hamiltonian                                                                            " begin include("test_hamiltonian.jl") end
    @safetestset "Lagrangian                                                                             " begin include("test_lagrangian.jl") end
    @safetestset "SymbolicNeuralNetwork                                                                  " begin include("test_symbolic_neuralnet.jl") end

end
