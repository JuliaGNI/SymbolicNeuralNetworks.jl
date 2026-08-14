# `cse = true` is the default for all code generation (see `_build_nn_function`). It changes
# *how* the expression is printed — a `let` block of shared bindings instead of a fully
# inlined tree — but must never change *what* is computed. These tests pin that down for
# every expression shape the package generates code for: plain forward passes, derivatives
# with respect to the input (`Jacobian`), derivatives with respect to the parameters
# (`Gradient`), the `NamedTuple`-valued builders, and the full `SymbolicPullback`.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, FeedForwardLoss
using Test
import Random

Random.seed!(123)

# deep enough that CSE actually has something to share
c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)

input = rand(3, 5)
output = rand(2, 5)

@testset "forward pass" begin
    eq = c(snn.input, params(snn))
    @test build_nn_function(eq, snn)(input, ps) ≈
          build_nn_function(eq, snn; cse = false)(input, ps)
end

@testset "Jacobian (derivative w.r.t. the input)" begin
    eq = derivative(Jacobian(snn))
    @test build_nn_function(eq, snn)(input, ps) ≈
          build_nn_function(eq, snn; cse = false)(input, ps)
end

@testset "Gradient (derivative w.r.t. the parameters)" begin
    eq = derivative(Gradient(snn))[1].L1.W
    @test build_nn_function(eq, snn)(input, ps) ≈
          build_nn_function(eq, snn; cse = false)(input, ps)
end

@testset "NamedTuple-valued equations" begin
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
    with_cse = build_nn_function(eqs, params(snn), snn.input)(input, ps)
    without_cse = build_nn_function(eqs, params(snn), snn.input; cse = false)(input, ps)
    @test with_cse.a ≈ without_cse.a
    @test with_cse.b ≈ without_cse.b
end

@testset "SymbolicPullback" begin
    loss = FeedForwardLoss()
    with_cse = SymbolicPullback(snn, loss)(ps, c, (input, output))[2](1.0)
    without_cse = SymbolicPullback(snn, loss; cse = false)(ps, c, (input, output))[2](1.0)
    for layer in keys(with_cse)
        @test with_cse[layer].W ≈ without_cse[layer].W
        @test with_cse[layer].b ≈ without_cse[layer].b
    end
end
