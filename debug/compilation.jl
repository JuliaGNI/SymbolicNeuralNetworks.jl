using RuntimeGeneratedFunctions
using SnoopCompile
using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics

# Creation of the HamiltonianNeuralNetwork
arch = HamiltonianNeuralNetwork(2; nhidden = 1, width = 4)
hnn = NeuralNetwork(arch, Float16)

@variables sx[1:2]
@variables nn(sx)[1:1]
Dx1 = Differential(sx[1])
Dx2 = Differential(sx[2])
vectorfield = [0 1; -1 0] * [Dx1(nn[1]), Dx2(nn[1])]
eqs = (x = sx, nn = nn, vectorfield = vectorfield)

shnn = SymbolicNeuralNetwork(arch; eqs = eqs)

fun = functions(shnn).vectorfield

x = [1, 2]
p = hnn.params

RuntimeGeneratedFunctions.init(@__MODULE__)

@time fun(x, p)




