using SymbolicNeuralNetworks: _build_nn_function
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using AbstractNeuralNetworks: params
using Test

# this tests the function '_build_nn_function' (not 'build_nn_function')
function apply_build_function(input_dim::Integer=2, output_dim::Integer=1, num_examples::Integer=3)
    c = Chain(Dense(input_dim, output_dim, tanh))
    nn = NeuralNetwork(c)
    snn = SymbolicNeuralNetwork(nn)
    eq = c(snn.input, params(snn))
    built_function = _build_nn_function(eq, params(snn), snn.input)
    input = rand(input_dim, num_examples)

    @test all(i -> (built_function(input, params(nn), i) ≈ c(input[:, i], params(nn))), 1:num_examples)
end

for input_dim ∈ (2, 3)
    for output_dim ∈ (1, 2)
        for num_examples ∈ (1, 2, 3)
            apply_build_function(input_dim, output_dim, num_examples)
        end
    end
end