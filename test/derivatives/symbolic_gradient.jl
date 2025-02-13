using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_differentials, symbolic_derivative, _build_nn_function
using LinearAlgebra: norm
using Symbolics, AbstractNeuralNetworks
using AbstractNeuralNetworks: NeuralNetworkParameters, params

import Zygote
import Random
Random.seed!(123)

"""
This test checks if we perform the parallelization in the correct way.
"""
function test_symbolic_gradient(input_dim::Integer = 3, output_dim::Integer = 1, hidden_dim::Integer = 2, T::DataType = Float64, second_dim::Integer = 3)
    @assert second_dim > 1 "second_dim must be greater than 1!"
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    sout = norm(c(snn.input, params(snn))) ^ 2
    sdparams = symbolic_differentials(params(snn))
    _sgrad = symbolic_derivative(sout, sdparams)
    input = rand(T, input_dim, second_dim)
    for k in 1:second_dim
        zgrad = Zygote.gradient(ps -> (norm(c(input[:, k], ps)) ^ 2), params(nn))[1].params
        for key1 in keys(_sgrad)
            for key2 in keys(_sgrad[key1])
                executable_gradient = _build_nn_function(_sgrad[key1][key2], params(snn), snn.input)
                sgrad = executable_gradient(input, params(nn), k)
                @test sgrad ≈ zgrad[key1][key2]
            end
        end
    end
    nothing
end

"""
Also checks the parallelization, but for the full function.
"""
function test_symbolic_gradient2(input_dim::Integer = 3, output_dim::Integer = 1, hidden_dim::Integer = 2, T::DataType = Float64, second_dim::Integer = 1, third_dim::Integer = 1)
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    nn = NeuralNetwork(c, T)
    snn = SymbolicNeuralNetwork(nn)
    sout = norm(c(snn.input, params(snn))) ^ 2
    input = rand(T, input_dim, second_dim, third_dim)
    zgrad = Zygote.gradient(ps -> (norm(c(input, ps)) ^ 2), params(nn))[1].params
    sdparams = symbolic_differentials(params(snn))
    _sgrad = symbolic_derivative(sout, sdparams)
    sgrad = build_nn_function(_sgrad, sparams, sinput)(input, params(nn))
    for key1 in keys(sgrad) for key2 in keys(sgrad[key1]) @test zgrad[key1][key2] ≈ sgrad[key1][key2] end end
end

for second_dim in (2, 3, 4)
    test_symbolic_gradient(3, 1, 2, Float64, second_dim)
end

# for (second_dim, third_dim) in ((1, 1), )
#     test_symbolic_gradient2(3, 1, 2, Float64, second_dim, third_dim)
# end