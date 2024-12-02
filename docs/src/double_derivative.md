# Double Derivative

We can easily differentiate a neural network twice by using [`SymbolicNeuralNetworks.Jacobian`](@ref) and [`SymbolicNeuralNetworks.Gradient`](@ref) together. We first use [`SymbolicNeuralNetworks.Jacobian`](@ref) to differentiate the network output with respect to its input:

```@example jacobian_gradient
using AbstractNeuralNetworks
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative
using Latexify: latexify

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
□ = Jacobian(nn)
# we show the derivative with respect to 
derivative(□) |> latexify
```

We see that the output is a matrix of size ``\mathrm{output\_dim} \times \mathrm{input\_dim}``. We can further compute the gradients of all entries of this matrix with [`SymbolicNeuralNetworks.Gradient`](@ref):

```@example jacobian_gradient
g = Gradient(derivative(□), nn)
nothing # hide
```

So [`SymbolicNeuralNetworks.Gradient`](@ref) differentiates every element of the matrix with respect to all neural network parameters. In order to access the gradient of the first element of the neural network with respect to the weight `b` in the first layer, we write:

```@example jacobian_gradient
matrix_index = (1, 1)
layer = :L1
weight = :b
derivative(g)[matrix_index...][layer][weight] |> latexify
```

If we now want to obtain an executable `Julia` function we have to use [`build_nn_function`](@ref). We call this function on:

```math
x = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
```

```@example jacobian_gradient
built_function = build_nn_function(derivative(g), nn.params, nn.input)

x = [1., 0.]
ps = NeuralNetworkParameters((L1 = (W = [1. 0.; 0. 1.], b = [0., 0.]), ))
built_function(x, ps)[matrix_index...][layer][weight]
```

!!! info
    With `SymbolicNeuralNetworks`, the `struct`s [`SymbolicNeuralNetworks.Jacobian`](@ref), [`SymbolicNeuralNetworks.Gradient`](@ref) and [`build_nn_function`](@ref) it is easy to build combinations of derivatives. This is much harder when using `Zygote`-based AD.