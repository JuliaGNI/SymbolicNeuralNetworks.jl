# The public entry points of `build_nn_function`, and the inputs it has to accept or reject.

using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Test
import Random

Random.seed!(123)

c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
input = rand(2, 4)

@testset "the network and the explicit form agree" begin
    eq = c(snn.input, params(snn))
    @test build_nn_function(eq, snn)(input, ps) ≈
          build_nn_function(eq, params(snn), snn.input)(input, ps)

    soutput = Symbolics.variables(:y, 1:2)
    two_input_eq = (c(snn.input, params(snn)) - soutput) .^ 2
    output = rand(2, 4)
    @test build_nn_function(two_input_eq, snn, soutput)(input, output, ps) ≈
          build_nn_function(two_input_eq, params(snn), snn.input, soutput)(input, output, ps)
end

# The generated code used to be assembled by matching the *name* `sinput` in its printed form, so
# any other name failed with an `AssertionError`. Arguments are matched by position now.
@testset "the symbolic variables may be named anything" begin
    for name in (:whatever, :sinput, :input, :ps_like)
        variables = Symbolics.variables(name, 1:2)
        f = build_nn_function(c(variables, params(snn)), params(snn), variables)
        @test f(input, ps) ≈ build_nn_function(c(snn.input, params(snn)), snn)(input, ps)
    end
end

# `@variables x[1:n]` produces a `Symbolics.Arr`, which `Symbolics.build_function` cannot generate
# code for. Passing one is still supported; it is scalarised on the way in.
@testset "Symbolics.Arr variables are scalarised" begin
    @variables z[1:2]
    f = build_nn_function(c(collect(z), params(snn)), params(snn), z)
    @test f(input, ps) ≈ build_nn_function(c(snn.input, params(snn)), snn)(input, ps)
end

# Reductions used to have to be written over `collect(...)`: an un-scalarised reduction produced an
# `arrayop`, whose generated code referred to variables that nothing bound.
@testset "reductions over the network output" begin
    for eq in (sum(c(snn.input, params(snn))), sum(collect(c(snn.input, params(snn)))))
        f = build_nn_function(eq, snn)
        @test vec(f(input, ps)) ≈ [sum(c(input[:, k], ps)) for k in axes(input, 2)]
    end
end

@testset "keyword arguments are validated" begin
    eq = c(snn.input, params(snn))
    @test_throws ArgumentError build_nn_function(eq, snn; reduce = *)
    @test_throws ArgumentError build_nn_function(eq, snn; reduce = vcat)
    @test build_nn_function(eq, snn; reduce = hcat) isa Function
    @test build_nn_function(eq, snn; reduce = +) isa Function
end

# A bare layer is wrapped in a `Chain`, so its parameters are nested like everybody else's.
@testset "a single layer can be used as the model" begin
    d = Dense(2, 1, tanh)
    single = SymbolicNeuralNetwork(d)
    single_nn = NeuralNetwork(single.model)
    @test keys(params(single)) == (:L1,)
    f = build_nn_function(single.model(single.input, params(single)), single)
    @test f(input, params(single_nn)) ≈
          reduce(hcat, [single.model(input[:, k], params(single_nn))
                        for k in axes(input, 2)])
end
