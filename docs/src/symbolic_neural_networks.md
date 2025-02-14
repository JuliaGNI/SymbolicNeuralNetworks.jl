# Symbolic Neural Networks

When using a symbolic neural network we can use *architectures* from [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) or more simple building blocks.

We first call the symbolic neural network that only consists of one layer:

```@example snn
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, params

input_dim = 2
output_dim = 1
hidden_dim = 3
c = Chain(Dense(input_dim, hidden_dim), Dense(hidden_dim, hidden_dim), Dense(hidden_dim, output_dim))
nn = SymbolicNeuralNetwork(c)
nothing # hide
```

We can now build symbolic expressions based on this neural network. Here we do so by calling `evaluate_equations`:

```@example snn
using Symbolics
using Latexify: latexify

@variables sinput[1:input_dim]
soutput = nn.model(sinput, params(nn))

soutput
```

or use `Symbolics.scalarize` to get a more readable version of the equation:

```@example snn
soutput |> Symbolics.scalarize
```

We can compute the symbolic gradient with [`SymbolicNeuralNetworks.Gradient`](@ref):

```@example snn
using SymbolicNeuralNetworks: derivative
derivative(SymbolicNeuralNetworks.Gradient(soutput, nn))[1].L1.b
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
surface!(x_vec, y_vec, [f([x, y]) for x in x_vec, y in y_vec]; alpha = .8, transparency = true)
fig
```

We now train the network:

```@example snn
import Random # hide
Random.seed!(123) # hide
nn_cpu = NeuralNetwork(c, CPU())
o = Optimizer(AdamOptimizer(), nn_cpu)
n_epochs = 1000
batch = Batch(10)
o(nn_cpu, dl, batch, n_epochs, pb.loss, pb; show_progress = false); # hide
@time o(nn_cpu, dl, batch, n_epochs, pb.loss, pb; show_progress = false);
nothing # hide
```

We now compare the neural network-approximated curve to the original one:

```@example snn
fig = Figure()
ax = Axis3(fig[1, 1])

surface!(x_vec, y_vec, [c([x, y], params(nn_cpu))[1] for x in x_vec, y in y_vec]; alpha = .8, colormap = :darkterrain, transparency = true)
fig
```

We can also compare the time it takes to train the [`SymbolicNeuralNetwork`](@ref) to the time it takes to train a *standard neural network*:

```@example snn
loss = FeedForwardLoss()
pb2 = GeometricMachineLearning.ZygotePullback(loss)
o(nn_cpu, dl, batch, n_epochs, pb2.loss, pb2; show_progress = false); # hide
@time o(nn_cpu, dl, batch, n_epochs, pb2.loss, pb2; show_progress = false);
nothing # hide
```

!!! info
    For the case presented here we do not observe speed-ups of the symbolic neural network over the standard neural network. For other cases, especially Hamiltonian neural networks, this is however different.