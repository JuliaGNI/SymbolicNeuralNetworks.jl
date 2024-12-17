using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: _get_params, _get_contents
using AbstractNeuralNetworks
using AbstractNeuralNetworks: params
using Symbolics
using GeometricMachineLearning: ZygotePullback
using Test
import Random
Random.seed!(123)

compare_values(arr1::Array, arr2::Array) = @test arr1 ≈ arr2
function compare_values(nt1::NamedTuple, nt2::NamedTuple)
    @assert keys(nt1) == keys(nt2)
    NamedTuple{keys(nt1)}((compare_values(arr1, arr2) for (arr1, arr2) in zip(values(nt1), values(nt2))))
end

function compare_symbolic_pullback_to_zygote_pullback(input_dim::Integer, output_dim::Integer, second_dim::Integer=1)
    c = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    spb = SymbolicPullback(snn, loss)
    input_output = (rand(input_dim, second_dim), rand(output_dim, second_dim))
    loss_and_pullback = spb(params(nn), nn.model, input_output)
    # note that we apply the second argument to another input `1`
    pb_values = loss_and_pullback[2](1)

    zpb = ZygotePullback(loss)
    loss_and_pullback_zygote = zpb(params(nn), nn.model, input_output)
    pb_values_zygote = loss_and_pullback_zygote[2](1) |> _get_contents |> _get_params

    compare_values(pb_values, pb_values_zygote)
end

for input_dim ∈ (2, 3)
    for output_dim ∈ (1, 2)
        compare_symbolic_pullback_to_zygote_pullback(input_dim, output_dim)
    end
end