using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_parameter_gradient, output_dimension, PullbackFunction
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, FeedForwardLoss
using Symbolics
using Test
import Zygote, Random

Random.seed!(123)

# This used to be `GeometricMachineLearning.ZygotePullback`, which is the one line below. It is
# inlined here so that the test suite does not depend on `GeometricMachineLearning`: that package has
# a compat bound on `SymbolicNeuralNetworks`, so depending on it in the other direction means neither
# can be released without the other having been released first.
zygote_pullback(loss, ps, model, input_output::Tuple) = Zygote.pullback(p -> loss(model, p, input_output...), ps)

compare_values(arr1::Array, arr2::Array) = @test arr1 ≈ arr2
function compare_values(nt1::NamedTuple, nt2::NamedTuple)
    @test keys(nt1) == keys(nt2)
    for key in keys(nt1)
        compare_values(nt1[key], nt2[key])
    end
end

@testset "single sample: input_dim = $input_dim, output_dim = $output_dim" for
        input_dim in (2, 3), output_dim in (1, 2)

    c = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    pb = SymbolicPullback(snn, loss)

    input_output = (rand(input_dim, 1), rand(output_dim, 1))
    symbolic = pb(params(nn), nn.model, input_output)[2](1)
    zygote = zygote_pullback(loss, params(nn), nn.model, input_output)[2](1)[1]
    compare_values(symbolic, params(zygote))
end

# `SymbolicPullback` differentiates the loss of a *single* sample and sums the results over the batch
# (`reduce = +`), so it computes the gradient of the summed per-sample loss. For a loss that is not
# additive over the batch — `FeedForwardLoss` normalises by `norm(output)` over the whole batch —
# that is not the same as the gradient of the batched loss. This pins down which of the two it is.
@testset "a batch gives the gradient of the summed per-sample loss" begin
    c = Chain(Dense(3, 2, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    pb = SymbolicPullback(snn, loss)

    input, output = rand(3, 4), rand(2, 4)
    symbolic = pb(params(nn), nn.model, (input, output))[2](1)
    per_sample = Zygote.gradient(params(nn)) do p
        sum(loss(nn.model, p, input[:, k:k], output[:, k:k]) for k in axes(input, 2))
    end[1]
    compare_values(symbolic, params(per_sample))
end

@testset "the loss value is returned alongside the pullback" begin
    c = Chain(Dense(2, 1, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    pb = SymbolicPullback(snn, loss)

    input, output = rand(2, 1), rand(1, 1)
    value, pullback = pb(params(nn), nn.model, (input, output))
    @test value ≈ loss(nn.model, params(nn), input, output)
    @test pullback isa PullbackFunction
    # the output sensitivities are ignored, as the loss is scalar-valued
    @test pullback(1) == pullback(2.0)
end

@testset "the pullback agrees with building the gradient by hand" begin
    c = Chain(Dense(2, 1, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    input_output = (rand(2), rand(1))

    from_pullback = SymbolicPullback(snn, loss)(params(nn), nn.model, input_output)[2](1)

    soutput = Symbolics.variables(:y, 1:output_dimension(nn.model))
    gradient = symbolic_parameter_gradient(loss(nn.model, params(snn), snn.input, soutput), snn)
    by_hand = build_nn_function(gradient, params(snn), snn.input, soutput; reduce = +)(input_output..., params(nn))

    @test from_pullback == params(by_hand)
end
