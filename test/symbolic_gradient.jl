using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_differentials, symbolic_gradient, build_executable_gradient, _build_executable_gradient
using LinearAlgebra: norm
using Symbolics, AbstractNeuralNetworks
using AbstractNeuralNetworks: NeuralNetworkParameters

import Zygote
import Random
Random.seed!(123)

"""
This test checks if we perform the parallelization in the correct way.
"""
function test_symbolic_gradient(input_dim::Integer = 3, output_dim::Integer = 1, hidden_dim::Integer = 2, T::DataType = Float64, second_dim::Integer = 3)
    @assert second_dim > 1 "second_dim must be greater than 1!"
    c = Chain(Dense(input_dim, hidden_dim, tanh), Dense(hidden_dim, output_dim, tanh))
    sparams = symbolicparameters(c)
    ps = initialparameters(c, T) |> NeuralNetworkParameters
    @variables sinput[1:input_dim] soutput[1:output_dim]
    sout = norm(c(sinput, sparams)) ^ 2
    sdparams = symbolic_differentials(sparams)
    _sgrad = symbolic_gradient(sout, sdparams)
    input = rand(T, input_dim, second_dim)
    for k in 1:second_dim
        zgrad = Zygote.gradient(ps -> (norm(c(input[:, k], ps)) ^ 2), ps)[1].params
        for key1 in keys(_sgrad)
            for key2 in keys(_sgrad[key1])
                executable_gradient = _build_executable_gradient(_sgrad[key1][key2], sinput, soutput, sparams)
                sgrad = executable_gradient(input, zero(input), ps, k)
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
    sparams = symbolicparameters(c)
    ps = initialparameters(c, T) |> NeuralNetworkParameters
    @variables sinput[1:input_dim]
    sout = norm(c(sinput, sparams)) ^ 2
    input = rand(T, input_dim, second_dim, third_dim)
    zgrad = Zygote.gradient(ps -> (norm(c(input, ps)) ^ 2), ps)[1].params
    sdparams = symbolic_differentials(sparams)
    _sgrad = symbolic_gradient(sout, sdparams)
    # soutput is required!
    @variables soutput[1:1]
    sgrad = build_executable_gradient(_sgrad, sinput, soutput, sparams)(input, zero(input), ps)
    for key1 in keys(sgrad) for key2 in keys(sgrad[key1]) @test zgrad[key1][key2] ≈ sgrad[key1][key2] end end
end

for second_dim in (2, 3, 4)
    test_symbolic_gradient(3, 1, 2, Float64, second_dim)
end

for (second_dim, third_dim) in ((1, 1), )
    test_symbolic_gradient2(3, 1, 2, Float64, second_dim, third_dim)
end