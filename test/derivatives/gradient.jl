using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Gradient, derivative, symbolic_differentials,
                              symbolic_derivative,
                              symbolic_parameter_gradient, build_kernel
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: norm
using Test
import Zygote, Random

Random.seed!(123)

@testset "Gradient agrees with symbolic_parameter_gradient" begin
    c = Chain(Dense(2, 1, tanh))
    snn = SymbolicNeuralNetwork(c)
    g = Gradient(snn)
    @test isequal(derivative(g), symbolic_parameter_gradient(g.f, snn))
    @test derivative(g) isa AbstractArray{<:NetworkParameters}
end

# A scalar expression is differentiated into the parameter shape directly; only an array-valued one
# produces an array of those. This is what lets `SymbolicPullback` use the result without unwrapping.
@testset "a scalar expression gives one parameter-shaped gradient" begin
    c = Chain(Dense(2, 1, tanh))
    snn = SymbolicNeuralNetwork(c)
    scalar = symbolic_parameter_gradient(sum(c(snn.input, params(snn))), snn)
    @test scalar isa NetworkParameters
    @test size(scalar.L1.W) == size(params(snn).L1.W)
end

"""
The kernel evaluates the gradient for one sample of a batch; compare it against `Zygote` for each
sample separately, which is what checks that the batch index is threaded through correctly.
"""
function test_symbolic_gradient(
        input_dim = 3, output_dim = 1, hidden_dim = 2, T = Float64, batch_size = 3)
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(c)
    sout = norm(c(snn.input, params(snn))) ^ 2
    sgrad = symbolic_derivative(sout, symbolic_differentials(params(snn)))

    input = rand(T, input_dim, batch_size)
    for k in 1:batch_size
        zgrad = Zygote.gradient(p -> norm(c(input[:, k], p)) ^ 2, params(nn))[1]
        for layer in keys(sgrad), array in keys(sgrad[layer])

            kernel = build_kernel(sgrad[layer][array], params(snn), snn.input)
            @test kernel(input, params(nn), k) ≈ zgrad[layer][array]
        end
    end
end

@testset "the gradient kernel indexes the batch correctly, batch_size = $batch_size" for batch_size in (2, 3, 4)
    test_symbolic_gradient(3, 1, 2, Float64, batch_size)
end

# The gradient of a batch is the sum of the per-sample gradients, which is what `reduce = +` builds.
@testset "the summed gradient over a batch, input rank $rank" for rank in (2, 3)
    input_dim, hidden_dim, output_dim = 3, 2, 1
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(c)
    sgrad = symbolic_parameter_gradient(norm(c(snn.input, params(snn))) ^ 2, snn)
    f = build_nn_function(sgrad, params(snn), snn.input; reduce = +)

    # `AbstractNeuralNetworks.Dense` computes `W * x`, which has no method for a three-dimensional
    # `x` — see https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/40. The reference is
    # therefore assembled sample by sample rather than by calling the chain on the whole batch.
    input = rank == 2 ? rand(input_dim, 4) : rand(input_dim, 2, 2)
    samples = reshape(input, input_dim, :)
    reference = Zygote.gradient(
        p -> sum(norm(c(samples[:, k], p)) ^ 2
        for k in axes(samples, 2)), params(nn))[1]

    result = f(input, params(nn))
    for layer in keys(reference), array in keys(reference[layer])

        @test result[layer][array] ≈ reference[layer][array]
    end
end
