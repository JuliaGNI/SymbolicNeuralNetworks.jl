using SymbolicNeuralNetworks: build_nn_function, SymbolicNeuralNetwork, function_valued_parameters
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork
using Test
import Random
Random.seed!(123)

function build_function_for_array_valued_equation(input_dim::Integer=2, output_dim::Integer=1)
    ch = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(ch)
    snn = SymbolicNeuralNetwork(nn)
    eqs = [(a = ch(snn.input, snn.params), b = ch(snn.input, snn.params).^2), (c = ch(snn.input, snn.params).^3, )]
    funcs = build_nn_function(eqs, snn.params, snn.input)
    input = Vector(1:input_dim)
    a = ch(input, nn.params)
    b = ch(input, nn.params).^2
    c = ch(input, nn.params).^3
    funcs_evaluated = funcs(input, nn.params)
    funcs_evaluated_as_vector = [funcs_evaluated[1].a, funcs_evaluated[1].b, funcs_evaluated[2].c]
    result_of_standard_computation = [a, b, c]

    @test funcs_evaluated_as_vector ≈ result_of_standard_computation
end

function build_function_for_named_tuple(input_dim::Integer=2, output_dim::Integer=1)
    c = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    eqs = (a = c(snn.input, snn.params), b = c(snn.input, snn.params).^2)
    funcs = build_nn_function(eqs, snn.params, snn.input)
    input = Vector(1:input_dim)
    a = c(input, nn.params)
    b = c(input, nn.params).^2
    funcs_evaluated = funcs(input, nn.params)

    funcs_evaluated_as_vector = [funcs_evaluated.a, funcs_evaluated.b]
    result_of_standard_computation = [a, b]

    @test funcs_evaluated_as_vector ≈ result_of_standard_computation
end

function function_valued_parameters_for_named_tuple(input_dim::Integer=2, output_dim::Integer=1)
    c = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    eqs = (a = c(snn.input, snn.params), b = c(snn.input, snn.params).^2)
    funcs = function_valued_parameters(eqs, snn.params, snn.input)
    input = Vector(1:input_dim)
    a = c(input, nn.params)
    b = c(input, nn.params).^2

    funcs_evaluated_as_vector = [funcs.a(input, nn.params), funcs.b(input, nn.params)]
    result_of_standard_computation = [a, b]
    
    @test funcs_evaluated_as_vector ≈ result_of_standard_computation
end

# we test in the following order: `function_valued_parameters` → `build_function` (for `NamedTuple`) → `build_function` (for `Array` of `NamedTuple`s) as this is also how the functions are built.
for input_dim ∈ (2, 3)
    for output_dim ∈ (1, 2)
        function_valued_parameters_for_named_tuple(input_dim, output_dim)
        build_function_for_named_tuple(input_dim, output_dim)
        build_function_for_array_valued_equation(input_dim, output_dim)
    end
end