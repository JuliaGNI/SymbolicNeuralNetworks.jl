using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: input_dimension, output_dimension
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, NeuralNetworkParameters,
                              UnknownArchitecture
using Symbolics
using Test

c = Chain(Dense(2, 3, tanh), Dense(3, 1, tanh))

@testset "the constructors agree" begin
    from_model = SymbolicNeuralNetwork(c)
    from_network = SymbolicNeuralNetwork(NeuralNetwork(c))
    from_architecture = SymbolicNeuralNetwork(UnknownArchitecture(), c)

    for snn in (from_model, from_network, from_architecture)
        @test snn.model === c
        @test length(snn.input) == input_dimension(c)
        @test snn.input isa Vector{Num}
        @test params(snn) isa NeuralNetworkParameters
        @test keys(params(snn)) == (:L1, :L2)
        @test size(params(snn).L1.W) == (3, 2)
        @test size(params(snn).L1.b) == (3,)
    end
end

@testset "dimensions" begin
    @test input_dimension(c) == 2
    @test output_dimension(c) == 1
    @test input_dimension(Dense(4, 7)) == 4
    @test output_dimension(Dense(4, 7)) == 7
end

@testset "the symbolic output has the shape of the numeric one" begin
    snn = SymbolicNeuralNetwork(c)
    soutput = c(snn.input, params(snn))
    @test soutput isa Vector{Num}
    @test length(soutput) == output_dimension(c)
end

@testset "show" begin
    @test occursin("SymbolicNeuralNetwork", sprint(show, SymbolicNeuralNetwork(c)))
end
