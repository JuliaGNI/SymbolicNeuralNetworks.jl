using SymbolicNeuralNetworks
using SafeTestsets
using Test

@testset "SymbolicNeuralNetworks.jl" begin
   
    @safetestset "Symbolic Params                                                                        " begin include("test_params.jl") end
    @safetestset "SymbolicNeuralNetwork                                                                  " begin include("test_symbolic_neuralnet.jl") end

end
