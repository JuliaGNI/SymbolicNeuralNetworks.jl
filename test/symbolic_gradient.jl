using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_differentials, symbolic_gradient, build_executable_gradient
using LinearAlgebra: norm
using Symbolics, AbstractNeuralNetworks
using AbstractNeuralNetworks: NeuralNetworkParameters

import Zygote
import Random
Random.seed!(123)

function test_symbolic_gradient(input_dim::Integer = 3, output_dim::Integer = 1, hidden_dim::Integer = 2, T::DataType = Float64, second_dim::Integer = 2, third_dim::Integer = 2)
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    sparams = symbolicparameters(c)
    ps = initialparameters(c, T) |> NeuralNetworkParameters
    @variables sinput[1:input_dim]
    soutput = norm(c(sinput, sparams))
    input = rand(T, input_dim)
    zgrad = Zygote.gradient(ps -> norm(c(input, ps)), ps)[1].params
    sdparams = symbolic_differentials(sparams)
    _sgrad = symbolic_gradient(soutput, sdparams)
    @variables empty[1:1]
    sgrad = build_executable_gradient(_sgrad, sinput, empty, sparams)(input, zero(input), ps)
    for key in keys(s_grad) @test zgrad[key] ≈ sgrad[key] end
end

test_symbolic_gradient()