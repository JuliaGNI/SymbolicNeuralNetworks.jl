# Symbolic Neural Networks

When using a symbolic neural network we can use *architectures* from [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) or more simple building blocks.

We first call the symbolic neural network that only consists of one layer:

```@example snn
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, initialparameters

input_dim = 2
output_dim = 1
c = Chain(Dense(input_dim, output_dim))
nn = SymbolicNeuralNetwork(c)
nothing # hide
```

We can now build symbolic expressions based on this neural network. Here we do so by calling `evaluate_equations`:

```@example snn
using Symbolics
using Latexify: latexify

@variables sinput[1:input_dim]
soutput = nn.model(sinput, nn.params)

soutput |> latexify
```

We can compute the symbolic gradient with [`SymbolicNeuralNetworks.Gradient`](@ref):

```@example snn
using SymbolicNeuralNetworks: derivative
derivative(SymbolicNeuralNetworks.Gradient(soutput, nn))[1].L1.b |> latexify
```

!!! info
    [`SymbolicNeuralNetworks.Gradient`](@ref) can also be called as `SymbolicNeuralNetworks.Gradient(snn)`, so without providing a specific output. In this case `soutput` is taken to be the symbolic output of the network (i.e. is equivalent to the construction presented here). Also note that we further called [`SymbolicNeuralNetworks.derivative`](@ref) here in order to get the symbolic gradient (as opposed to the symbolic output of the neural network).

In order to *train* a [`SymbolicNeuralNetwork`](@ref) we can use:

```@example snn
pb = SymbolicPullback(nn)

nothing # hide
```