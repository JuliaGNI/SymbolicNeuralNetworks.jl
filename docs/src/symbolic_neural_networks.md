# Symbolic Neural Networks

When using a symbolic neural network we can use *architectures* from [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) or more simple building blocks.

We first call the symbolic neural network that only consists of one layer:

```julia snn
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Dense, initialparameters

input_dim = 2
d = Dense(input_dim, 1, tanh)
nn = SymbolicNeuralNetwork(d)
```

We can now build symbolic expressions based on this neural network. Here we do so by calling `evaluate_equations`:

```julia snn
using Symbolics
import Latexify

@variables x[1:input_dim] ∇nn[1:input_dim] output[1:1]
eqs = (x = x, nn = output, ∇nn = ∇nn)
evaluated_equations = evaluate_equations(nn, eqs = eqs)

evaluated_equations.∇nn |> Latexify.latexify
```

Equivalently we can also use [`SymbolicNeuralNetworks.Jacobian`](@ref):

```julia snn
evaluated_equations_alternative = gradient(d)

evaluated_equations_alternative |> Latexify.latexify
```