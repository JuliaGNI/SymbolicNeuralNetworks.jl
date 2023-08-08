using SymbolicNeuralNetworks
using GeometricMachineLearning

using Test

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2), Float64)
shnn = Symbolize(hnn, 2)

@test typeof(shnn) <: SymbolicNeuralNetwork{<:HamiltonianNeuralNetwork}
#@test architecture(shnn) == hnn.architecture
@test params(shnn) == hnn.params
@test model(shnn) == hnn.model

x = [0.5, 0.8]
@test shnn(x) == hnn(x)
@time shnn(x)
@time hnn(x)


