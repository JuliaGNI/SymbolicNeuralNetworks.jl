using SymbolicNeuralNetworks: build_nn_function, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, initialparameters, NeuralNetworkParameters
using Test
import Random
Random.seed!(123)

function build_function_for_array_valued_equation(input_dim::Integer=2, output_dim::Integer=1)
    ch = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(ch)
    snn = SymbolicNeuralNetwork(nn)
    eqs = [(a = ch(snn.input, snn.params), b = ch(snn.input, snn.params).^2), (c = ch(snn.input, snn.params).^3, )]
    funcs = build_nn_function(eqs, nn.params, nn.input)
    input = [1., 2.]
    a = ch(input, ps)
    b = ch(input, ps).^2
    c = ch(input, ps).^3
    funcs_evaluated = funcs(input, ps)
    funcs_evaluated_as_vector = [funcs_evaluated[1].a, funcs_evaluated[1].b, funcs_evaluated[2].c]
    result_of_standard_computation = [a, b, c]

    @test funcs_evaluated_as_vector ≈ result_of_standard_computation
end

for input_dim ∈ (2, 3)
    for output_dim ∈ (1, 2)
        build_function_for_array_valued_equation(input_dim, output_dim)
    end
end