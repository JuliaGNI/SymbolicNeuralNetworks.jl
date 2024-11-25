using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using GeometricMachineLearning: ZygotePullback
using Symbolics
using Test
using Zygote

function test_hnn_loss(input_dim::Integer = 2, nhidden::Integer = 1, hidden_dim::Integer = 3, T::DataType = Float64, second_axis = 1, third_axis = 1)
    c = Chain(Dense(input_dim, hidden_dim, tanh), Tuple(Dense(hidden_dim, hidden_dim, tanh) for _ in 1:nhidden)..., Dense(hidden_dim, 1, identity; use_bias = false))
    nn = NeuralNetwork(c, T)
    snn = HamiltonianSymbolicNeuralNetwork(c)
    loss = HNNLoss(snn)
    zpb = ZygotePullback(loss)
    spb = SymbolicPullback(snn)
    input = rand(T, input_dim, second_axis, third_axis)
    output = rand(T, input_dim, second_axis, third_axis)
    # the x -> x[1].params is necessary because of Zygote idiosyncrasies
    zpb_evaluated = zpb(nn.params, c, (input, output))[2](1)[1].params
    spb_evaluated = spb(nn.params, c, (input, output))[2](1)
    @assert keys(zpb_evaluated) == keys(spb_evaluated)
    for key in keys(zpb_evaluated) @assert keys(zpb_evaluated[key]) == keys(spb_evaluated[key]) end
    for key1 in keys(zpb_evaluated) for key2 in keys(zpb_evaluated[key1]) @test zpb_evaluated[key1][key2] ≈ spb_evaluated[key1][key2] end end
end

for input_dim in (2, 3)
    for nhidden in (1, )
        for hidden_dim in (3, 5)
            for T in (Float32, Float64)
                test_hnn_loss(input_dim, nhidden, hidden_dim, T)
            end
        end
    end
end