using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using Test
using Zygote

# Creation of SymbolicNeuralNetwork

# with eqs

@variables sx[1:2]
@variables nn(sx)[1:1]
Dx1 = Differential(sx[1])
Dx2 = Differential(sx[2])
eq = [0 1; -1 0] * [Dx1(nn[1]), Dx2(nn[1])]

eqs = (x = sx, nn = nn, eq1 = eq)

arch = HamiltonianNeuralNetwork(2)
hnn = NeuralNetwork(arch, Float64)

shnn = SymbolicNeuralNetwork(arch; eqs = eqs)

@test typeof(shnn) <: SymbolicNeuralNetwork{<:HamiltonianNeuralNetwork}
@test architecture(shnn) == arch
@test model(shnn)   == hnn.model
@test params(shnn) === symbolize(hnn.params)[1]
@test keys(equations(shnn)) == (:eq1, :eval)
@test keys(functions(shnn)) == (:eq1, :eval)

x = [0.5, 0.8]

println("Compareason of performances between an clasical neuralnetwork and a symbolic one")
@time hnn(x)
@time shnn(x, hnn.params)
@test shnn(x, hnn.params) == hnn(x, hnn.params)

fun_eq1 = functions(shnn).eq1

∇ₓnn(x, params) = Zygote.gradient(x->hnn(x, params)[1], x)[1]

@test_nowarn fun_eq1(x, hnn.params)
@test_nowarn ∇ₓnn(x, hnn.params)[1]

println("Compareason of performances between Zygote and SymbolicNeuralNetwork for ∇ₓnn")
@time ∇ₓnn(x, hnn.params)[1]
@time fun_eq1(x, hnn.params)
@test fun_eq1(x, hnn.params) == ∇ₓnn(x, hnn.params)[1]

# with dim

shnn2 = SymbolicNeuralNetwork(arch, 2)

@test typeof(shnn) <: SymbolicNeuralNetwork{<:HamiltonianNeuralNetwork}
@test architecture(shnn) == arch
@test model(shnn)   == hnn.model
@test params(shnn) === symbolize(hnn.params)[1]
@test keys(equations(shnn)) == (:eq1, :eval)
@test keys(functions(shnn)) == (:eq1, :eval)

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