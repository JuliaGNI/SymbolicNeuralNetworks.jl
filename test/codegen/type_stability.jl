# The built functions used to be local closures with several methods, whose type could not be named
# and whose return type Julia could not always infer. They are named `struct`s now (see
# `src/codegen/batched_function.jl`), which makes them inferable and lets a caller store one in a
# typed field. These tests pin that down for every call shape.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative, AbstractBatchedFunction
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, FeedForwardLoss
using Symbolics
using Test
import Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
soutput = Symbolics.variables(:y, 1:2)

@testset "one data argument, reduce = $reduction, inplace = $inplace" for
        reduction in (hcat, +), inplace in (true, false)

    f = build_nn_function(c(snn.input, params(snn)), snn; reduce = reduction, inplace = inplace)
    @test f isa AbstractBatchedFunction
    @test isconcretetype(typeof(f))
    @inferred f(rand(3, 5), ps)
    @inferred f(rand(3), ps)
    @inferred f(rand(3, 2, 3), ps)
end

@testset "matrix-valued equations, reduce = $reduction" for reduction in (hcat, +)
    f = build_nn_function(derivative(Jacobian(snn)), snn; reduce = reduction)
    @inferred f(rand(3, 5), ps)
    @inferred f(rand(3), ps)
end

@testset "two data arguments, reduce = $reduction, inplace = $inplace" for
        reduction in (hcat, +), inplace in (true, false)

    f = build_nn_function((c(snn.input, params(snn)) - soutput) .^ 2, snn, soutput;
                          reduce = reduction, inplace = inplace)
    @inferred f(rand(3, 5), rand(2, 5), ps)
    @inferred f(rand(3), rand(2), ps)
end

@testset "equation sets" begin
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
    f = build_nn_function(eqs, params(snn), snn.input)
    @test isconcretetype(typeof(f))
    @inferred f(rand(3, 5), ps)
end

@testset "the pullback" begin
    pb = SymbolicPullback(snn, FeedForwardLoss())
    @test isconcretetype(typeof(pb))
    input, output = rand(3, 5), rand(2, 5)
    pullback = pb(ps, c, (input, output))[2]
    @test isconcretetype(typeof(pullback))
    @inferred pullback(1.0)
end
