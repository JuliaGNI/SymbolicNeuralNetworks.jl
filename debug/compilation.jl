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

code = shnn.code.vectorfield


test_code =  :(function (x::Vector, params::Tuple)
      begin
          ((((1 + -1 * tanh((((getindex((params[2]).W, 1, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 1, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 1, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 1, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 1)) ^ 2) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 2)) * getindex((params[2]).W, 1, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 2)) * getindex((params[2]).W, 1, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 2)) * getindex((params[2]).W, 1, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 2)) * getindex((params[2]).W, 1, 4))) * getindex((params[3]).W, 1, 1) + ((1 + -1 * tanh((((getindex((params[2]).W, 2, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 2, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 2, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 2, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 2)) ^ 2) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 2)) * getindex((params[2]).W, 2, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 2)) * getindex((params[2]).W, 2, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 2)) * getindex((params[2]).W, 2, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 2)) * getindex((params[2]).W, 2, 4))) * getindex((params[3]).W, 1, 2)) + ((1 + -1 * tanh((((getindex((params[2]).W, 3, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 3, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 3, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 3, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 3)) ^ 2) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 2)) * getindex((params[2]).W, 3, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 2)) * getindex((params[2]).W, 3, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 2)) * getindex((params[2]).W, 3, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 2)) * getindex((params[2]).W, 3, 4))) * getindex((params[3]).W, 1, 3)) + ((1 + -1 * tanh((((getindex((params[2]).W, 4, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 4, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 4, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 4, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 4)) ^ 2) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 2)) * getindex((params[2]).W, 4, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 2)) * getindex((params[2]).W, 4, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 2)) * getindex((params[2]).W, 4, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 2)) * getindex((params[2]).W, 4, 4))) * getindex((params[3]).W, 1, 4), ((((-1 * (1 + -1 * tanh((((getindex((params[2]).W, 1, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 1, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 1, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 1, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 1)) ^ 2)) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 1)) * getindex((params[2]).W, 1, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 1)) * getindex((params[2]).W, 1, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 1)) * getindex((params[2]).W, 1, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 1)) * getindex((params[2]).W, 1, 4))) * getindex((params[3]).W, 1, 1) + ((-1 * (1 + -1 * tanh((((getindex((params[2]).W, 2, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 2, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 2, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 2, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 2)) ^ 2)) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 1)) * getindex((params[2]).W, 2, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 1)) * getindex((params[2]).W, 2, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 1)) * getindex((params[2]).W, 2, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 1)) * getindex((params[2]).W, 2, 4))) * getindex((params[3]).W, 1, 2)) + ((-1 * (1 + -1 * tanh((((getindex((params[2]).W, 3, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 3, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 3, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 3, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 3)) ^ 2)) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 1)) * getindex((params[2]).W, 3, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 1)) * getindex((params[2]).W, 3, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 1)) * getindex((params[2]).W, 3, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 1)) * getindex((params[2]).W, 3, 4))) * getindex((params[3]).W, 1, 3)) + ((-1 * (1 + -1 * tanh((((getindex((params[2]).W, 4, 1) * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) + getindex((params[2]).W, 4, 2) * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2))) + getindex((params[2]).W, 4, 3) * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3))) + getindex((params[2]).W, 4, 4) * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4))) + getindex((params[2]).b, 4)) ^ 2)) * (((((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 1, 1) + getindex(x, 2) * getindex((params[1]).W, 1, 2)) + getindex((params[1]).b, 1)) ^ 2) * getindex((params[1]).W, 1, 1)) * getindex((params[2]).W, 4, 1) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 2, 1) + getindex(x, 2) * getindex((params[1]).W, 2, 2)) + getindex((params[1]).b, 2)) ^ 2) * getindex((params[1]).W, 2, 1)) * getindex((params[2]).W, 4, 2)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 3, 1) + getindex(x, 2) * getindex((params[1]).W, 3, 2)) + getindex((params[1]).b, 3)) ^ 2) * getindex((params[1]).W, 3, 1)) * getindex((params[2]).W, 4, 3)) + ((1 + -1 * tanh((getindex(x, 1) * getindex((params[1]).W, 4, 1) + getindex(x, 2) * getindex((params[1]).W, 4, 2)) + getindex((params[1]).b, 4)) ^ 2) * getindex((params[1]).W, 4, 1)) * getindex((params[2]).W, 4, 4))) * getindex((params[3]).W, 1, 4)
      end
  end)



RuntimeGeneratedFunctions.init(@__MODULE__)

test_fun = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(test_code))

@time fun(x, p)
@time test_fun(x,p)



