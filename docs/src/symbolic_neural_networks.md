# Symbolic Neural Networks

When using a symbolic neural network we can use *architectures* from [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) or more simple building blocks.

```@example
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Dense, initialparameters
import Symbolics

input_dim = 2
d = Dense(input_dim, 1, tanh)

x = Symbolics.variables(:x, 1:input_dim)
∇nn = Symbolics.variables(:∇nn, 1:input_dim)
nn = Symbolics.variables(:nn, 1:1)
Symbolics.@variables nn ∇nn
eqs = (x = x, nn = nn, ∇nn = ∇nn)
nn = SymbolicNeuralNetwork(d; eqs = eqs)

nn.equations.s∇output
```