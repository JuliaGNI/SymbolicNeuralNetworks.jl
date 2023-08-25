using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using Test
using Zygote


@variables sx[1:2]
@variables nn(sx)[1:1]
Dx1 = Differential(sx[1])
Dx2 = Differential(sx[2])
eq = [0 1; -1 0] * [Dx1(nn[1]), Dx2(nn[1])]

eqs = (x = sx, nn = nn, eq1 = eq)

arch = HamiltonianNeuralNetwork(2)
hnn = NeuralNetwork(arch, Float64)
shnn = SymbolicNeuralNetwork(arch; eqs = eqs)

@test keys(equations(shnn)) == (:eq1, :eval)
@test keys(functions(shnn)) == (:eq1, :eval)


fun_eq1 = functions(shnn).eq1

∇ₓnn(x, params) = Zygote.gradient(x->hnn(x, params)[1], x)[1]

@test_nowarn fun_eq1(x, hnn.params)
@test_nowarn ∇ₓnn(x, hnn.params)[1]

println("Compareason of performances between Zygote and SymbolicNeuralNetwork for ∇ₓnn")
@time ∇ₓnn(x, hnn.params)[1]
@time fun_eq1(x, hnn.params)
@test fun_eq1(x, hnn.params) == ∇ₓnn(x, hnn.params)[1]


