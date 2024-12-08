using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using Symbolics
using Test

function set_up_network()
    c = Chain(Dense(2, 3))
    nn = SymbolicNeuralNetwork(c)
    soutput = nn.model(nn.input, nn.params)
    nn_cpu = NeuralNetwork(c)
    nn, soutput, nn_cpu
end

function test_for_input()
    nn, soutput, nn_cpu = set_up_network()
    input = rand(2, 5)
    input2 = reshape((@view input[:, 1:2]), 2, 1, 2)
    built_function = build_nn_function(soutput, nn.params, nn.input)
    outputs = built_function(input2, nn_cpu.params)
    for i in 1:2
        @test nn.model(input[:, i], nn_cpu.params) ≈ outputs[:, 1, i]
    end
end

function test_for_input_and_output()
    nn, soutput2, nn_cpu = set_up_network()
    input = rand(2, 5)
    output = rand(3, 5)
    input2 = reshape((@view input[:, 1:2]), 2, 1, 2)
    output2 = reshape((@view output[:, 1:2]), 3, 1, 2)
    @variables soutput[1:3]
    built_function = build_nn_function((soutput - soutput2).^2, nn.params, nn.input, soutput)
    outputs = built_function(input2, output2, nn_cpu.params)
    for i in 1:2
        @test (nn.model(input[:, i], nn_cpu.params) - output[:, i]).^2 ≈ outputs[:, 1, i]
    end
end

test_for_input()
test_for_input_and_output()