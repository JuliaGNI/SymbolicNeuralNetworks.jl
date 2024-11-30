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

or use `Symbolics.scalarize` to get a more readable version of the equation:

```@example snn
soutput |> Symbolics.scalarize |> latexify
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

!!! info
    [`SymbolicNeuralNetworks.Gradient`](@ref) and [`SymbolicPullback`](@ref) both use the function [`SymbolicNeuralNetworks.symbolic_pullback`](@ref) internally, so are computationally equivalent. [`SymbolicPullback`](@ref) should however be used in connection to a `NetworkLoss` and [`SymbolicNeuralNetworks.Gradient`](@ref) can be used more generally to compute the derivative of array-valued expressions.

We want to use our one-layer neural network to approximate a Gaussian on the interval ``[-1, 1]\times[-1, 1]``. We fist generate the data for this task:

```@example snn
using GeometricMachineLearning

x_vec = -1.:.1:1.
y_vec = -1.:.1:1.
xy_data = hcat([[x, y] for x in x_vec, y in y_vec]...)
f(x::Vector) = exp.(-sum(x.^2))
z_data = mapreduce(i -> f(xy_data[:, i]), hcat, axes(xy_data, 2))

dl = DataLoader(xy_data, z_data)
nothing # hide
```

Note that we use `GeometricMachineLearning.DataLoader` to process the data. We further also visualize them: 

```@example snn
using CairoMakie

fig = Figure()
ax = Axis3(fig[1, 1])
surface!(x_vec, y_vec, [f([x, y]) for x in x_vec, y in y_vec]; alpha = .8)
fig
```

We now train the network:

```@example snn
import Random # hide
Random.seed!(123) # hide
nn_cpu = NeuralNetwork(c, CPU())
o = Optimizer(AdamOptimizer(), nn_cpu)
o(nn_cpu, dl, Batch(10), pb)
```