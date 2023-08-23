using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics

using Test

##################
@variables sx[1:1]
@variables nn

eq = Symbolics.gradient(nn, sx)

eqs = (x = sx, nn = nn)

arch = HamiltonianNeuralNetwork(1)
hnn = NeuralNetwork(arch, Float64)

shnn = SymbolicNeuralNetwork(arch; eqs = eqs)

@test typeof(shnn) <: SymbolicNeuralNetwork{<:HamiltonianNeuralNetwork}
@test architecture(shnn) == arch
@test model(shnn)   == hnn.model
@test params(shnn) === symbolic_params(hnn)
@test keys(equations(shnn)) == (:eval,)
@test keys(functions(shnn)) == (:eval,)

x = [0.5]

println("Compareason of performances between an clasical neuralnetwork and a symbolic one")
@time hnn(x)
@time shnn(x, hnn.params)
@test shnn(x, hnn.params) == hnn(x, hnn.params)



##################

hnn2  = symbolize(hnn; eqs = eqs)

@test typeof(hnn2.model) <: SymbolicModel
@test architecture(hnn2) == hnn.architecture
@test model(hnn2) == hnn.model
@test params(hnn2) == hnn.params

println("Compareason of performances between an clasical neuralnetwork and a symbolized one")
@time hnn(x)
@time hnn2(x)
