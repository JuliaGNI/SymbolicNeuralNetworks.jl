using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using Test
using Zygote
using Distances

@variables sx[1:2]
@variables nn(sx)[1:1]
Dx1 = Differential(sx[1])
Dx2 = Differential(sx[2])
vectorfield = [0 1; -1 0] * [Dx1(nn[1]), Dx2(nn[1])]

eqs = (x = sx, nn = nn, vectorfield = vectorfield)

arch = HamiltonianNeuralNetwork(2)
shnn = SymbolicNeuralNetwork(arch; eqs = eqs)

@test keys(equations(shnn)) == (:vectorfield, :eval)
@test keys(functions(shnn)) == (:vectorfield, :eval)

x = [0.5, 0.8]

hnn = NeuralNetwork(arch, Float64)
fun_vectorfield = functions(shnn).vectorfield
ω∇ₓnn(x, params) = [0 1; -1 0] * Zygote.gradient(x->hnn(x, params)[1], x)[1]

@test_nowarn fun_vectorfield(x, hnn.params)
@test_nowarn ω∇ₓnn(x, hnn.params)

println("Comparison of performances between Zygote and SymbolicNeuralNetwork for ω∇ₓnn")
@time ω∇ₓnn(x, hnn.params)
@time fun_vectorfield(x, hnn.params)
@test chebyshev(fun_vectorfield(x, hnn.params),ω∇ₓnn(x, hnn.params)) < 1e-15


