using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using LinearAlgebra

using Test

# Creation of SymbolicNeuralNetwork

@variables sx[1:2]
@variables nn

eq = Symbolics.gradient(nn, sx)

eqs = (x = sx, nn = nn)

arch = HamiltonianNeuralNetwork(2)
hnn = NeuralNetwork(arch, Float64)

shnn = SymbolicNeuralNetwork(arch; eqs = eqs)

@test typeof(shnn) <: SymbolicNeuralNetwork{<:HamiltonianNeuralNetwork}
@test architecture(shnn) == arch
@test model(shnn)   == hnn.model
@test params(shnn) === symbolic_params(hnn)
@test keys(equations(shnn)) == (:eval,)
@test keys(functions(shnn)) == (:eval,)

x = [0.5, 0.8]

println("Compareason of performances between an clasical neuralnetwork and a symbolic one")
@time hnn(x)
@time shnn(x, hnn.params)
@test shnn(x, hnn.params) == hnn(x, hnn.params)


shnn2 = SymbolicNeuralNetwork(arch, 2)

@test typeof(shnn) <: SymbolicNeuralNetwork{<:HamiltonianNeuralNetwork}
@test architecture(shnn) == arch
@test model(shnn)   == hnn.model
@test params(shnn) === symbolic_params(hnn)
@test keys(equations(shnn)) == (:eval,)
@test keys(functions(shnn)) == (:eval,)

@test shnn(x, hnn.params) == shnn2(x, hnn.params)

# Symbolisation of NeuralNetwork

hnns  = symbolize(hnn; eqs = eqs)

@test typeof(hnns.model) <: SymbolicModel
@test architecture(hnns) == hnn.architecture
@test model(hnns) == hnn.model
@test params(hnns) == hnn.params

println("Compareason of performances between an clasical neuralnetwork and a symbolized one")
@time hnn(x)
@time hnns(x)


hnns2  = symbolize(hnn, 2)

@test typeof(hnns2.model) <: SymbolicModel
@test architecture(hnns2) == hnn.architecture
@test model(hnns2) == hnn.model
@test params(hnns2) == hnn.params