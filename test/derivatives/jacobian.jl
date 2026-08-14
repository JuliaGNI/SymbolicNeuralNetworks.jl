using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Test
import ForwardDiff, Random

Random.seed!(123)

@doc raw"""
The gradient of a single dense layer ``d: x \mapsto \mathrm{tanh}(v^Tx + b)`` is
``\nabla{}d: x \mapsto \mathrm{tanh}'(v^Tx + b)v``; here it is compared against `ForwardDiff`.
"""
function test_jacobian(n::Integer, T = Float32)
    c = Chain(Dense(n, 1, tanh))
    snn = SymbolicNeuralNetwork(c)
    j = Jacobian(snn)

    ps = params(NeuralNetwork(c, T))
    input = rand(T, n)
    @test build_nn_function(j.f, snn)(input, ps) ≈ c(input, ps)
    @test build_nn_function(derivative(j), snn)(input, ps) ≈ ForwardDiff.jacobian(x -> c(x, ps), input)
end

@testset "single-layer Jacobian, n = $n, $T" for n in 1:10, T in (Float32, Float64)
    test_jacobian(n, T)
end

@testset "the convention is ∂fᵢ/∂xⱼ" begin
    c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
    snn = SymbolicNeuralNetwork(c)
    @test size(derivative(Jacobian(snn))) == (2, 3)      # output_dim × input_dim

    ps = params(NeuralNetwork(c, Float64))
    input = rand(3)
    @test build_nn_function(derivative(Jacobian(snn)), snn)(input, ps) ≈
          ForwardDiff.jacobian(x -> c(x, ps), input)
end

# The rows are indexed by `vec(f)`, so a scalar `f` gives a 1×n Jacobian — its gradient with respect
# to the input as a row. This used to be a `MethodError: no method matching vec(::Num)`.
@testset "a scalar expression" begin
    c = Chain(Dense(3, 2, tanh))
    snn = SymbolicNeuralNetwork(c)
    f = sum(c(snn.input, params(snn)))
    ps = params(NeuralNetwork(c, Float64))
    input = rand(3)

    j = derivative(Jacobian(f, snn))
    @test size(j) == (1, 3)
    @test build_nn_function(j, snn)(input, ps) ≈ ForwardDiff.gradient(x -> sum(c(x, ps)), input)'
end

@testset "an explicitly supplied expression" begin
    c = Chain(Dense(2, 2, tanh))
    snn = SymbolicNeuralNetwork(c)
    f = c(snn.input, params(snn)) .^ 2
    ps = params(NeuralNetwork(c, Float64))
    input = rand(2)
    @test build_nn_function(derivative(Jacobian(f, snn)), snn)(input, ps) ≈
          ForwardDiff.jacobian(x -> c(x, ps) .^ 2, input)
end
