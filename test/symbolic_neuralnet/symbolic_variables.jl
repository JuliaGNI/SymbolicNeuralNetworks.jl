using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_variables, symbolic_variables!, next_name!
using NeuralNetworkParameters: NetworkParameters
using Symbolics
using Test

@testset "next_name! counts per name" begin
    counters = Dict{Symbol, Int}()
    @test next_name!(counters, :var) === :var_1
    @test next_name!(counters, :var) === :var_2
    @test next_name!(counters, :other) === :other_1
    @test counters == Dict(:var => 2, :other => 1)
end

@testset "leaves are numbered in order, whatever their shape" begin
    counters = Dict{Symbol, Int}()
    scalar = symbolic_variables!(counters, 0.1, :X)
    vector = symbolic_variables!(counters, rand(2), :X)
    matrix = symbolic_variables!(counters, rand(2, 3), :X)

    @test scalar isa Num
    @test vector isa Vector{Num}
    @test matrix isa Matrix{Num}
    @test size(matrix) == (2, 3)
    @test counters[:X] == 3
    # the shapes are built from scalar variables, not from a `Symbolics.Arr`, so that they can be
    # differentiated with respect to entry by entry
    @test !(vector isa Symbolics.Arr)
end

# a tuple of `NamedTuple`s and `Tuple`s, including scalar entries
parameters = NetworkParameters((L1 = (W = [1, 1], b = [2, 2]),
                                L2 = (W = (a = 4, b = [1, 2]), c = 2),
                                L3 = [4 5; 7 8],
                                L4 = (a = 7, b = 8.2)))

@testset "nested parameter sets keep their nesting" begin
    sparams = symbolic_variables(parameters, :W)

    expected = NetworkParameters((L1 = (W = Symbolics.variables(:W_1, 1:2),
                                        b = Symbolics.variables(:W_2, 1:2)),
                                  L2 = (W = (a = Symbolics.variable(:W_3),
                                             b = Symbolics.variables(:W_4, 1:2)),
                                        c = Symbolics.variable(:W_5)),
                                  L3 = Symbolics.variables(:W_6, 1:2, 1:2),
                                  L4 = (a = Symbolics.variable(:W_7),
                                        b = Symbolics.variable(:W_8))))

    @test keys(sparams) == keys(expected)
    @test isequal(sparams.L1.W, expected.L1.W)
    @test isequal(sparams.L2.W.a, expected.L2.W.a)
    @test isequal(sparams.L2.c, expected.L2.c)
    @test isequal(sparams.L3, expected.L3)
    @test isequal(sparams.L4.b, expected.L4.b)
end

@testset "every call starts a fresh numbering" begin
    @test isequal(symbolic_variables(parameters, :W), symbolic_variables(parameters, :W))
end
