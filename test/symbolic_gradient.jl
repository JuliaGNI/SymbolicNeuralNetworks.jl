using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_differentials, symbolic_gradient, build_executable_gradient
using LinearAlgebra: norm
using Symbolics, AbstractNeuralNetworks
using AbstractNeuralNetworks: NeuralNetworkParameters

import Zygote
import Random
Random.seed!(123)

function test_symbolic_gradient(input_dim::Integer = 3, output_dim::Integer = 1, hidden_dim::Integer = 2, T::DataType = Float64, second_dim::Integer = 1, third_dim::Integer = 1)
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    sparams = symbolicparameters(c)
    ps = initialparameters(c, T) |> NeuralNetworkParameters
    @variables sinput[1:input_dim]
    sout = norm(c(sinput, sparams))
    input = rand(T, input_dim, second_dim, third_dim)
    zgrad = Zygote.gradient(ps -> norm(c(input, ps)), ps)[1].params
    sdparams = symbolic_differentials(sparams)
    _sgrad = symbolic_gradient(sout, sdparams)
    # soutput is required!
    @variables soutput[1:1]
    sgrad = build_executable_gradient(_sgrad, sinput, soutput, sparams)(input, zero(input), ps)
    for key1 in keys(sgrad) for key2 in keys(sgrad[key1]) @test zgrad[key1][key2] ≈ sgrad[key1][key2] end end
end

test_symbolic_gradient()