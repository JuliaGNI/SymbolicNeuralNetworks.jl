# Arbitrarily Combining Derivatives

`SymbolicNeuralNetworks` can compute derivatives of arbitrary order of a neural network. For this we use two `struct`s:
1. [`SymbolicNeuralNetworks.Jacobian`](@ref) and
2. [`SymbolicNeuralNetworks.Gradient`](@ref).

!!! info "Terminology"
    Whereas the name `Jacobian` is standard for the matrix whose entries consist of all partial derivatives of the output of a function, the name `Gradient` is typically not used the way it is done here. Normally a *gradient* collects all the partial derivatives of a scalar function. In `SymbolicNeuralNetworks` the `struct` `Gradient` performs all partial derivatives of a symbolic array with respect to all the parameters of a neural network. So if we compute the `Gradient` of a matrix, then the corresponding routine returns *a matrix of neural network parameters*, each of which is the *standard gradient* of a matrix element. So it can be written as:
    ```math
    \mathtt{Gradient}\left( \begin{pmatrix} m_{11} & m_{12} & \cdots & m_{1m} \\ m_{21} & m_{22} & \cdots & m_{2m} \\ \vdots & \vdots & \vdots & \vdots \\ m_{n1} & m_{n2} & \cdots & m_{nm} \end{pmatrix} \right) = \begin{pmatrix} \nabla_{\mathbb{P}}m_{11} & \nabla_{\mathbb{P}}m_{12} & \cdots & \nabla_{\mathbb{P}}m_{1m} \\ \nabla_{\mathbb{P}}m_{21} & \nabla_{\mathbb{P}}m_{22} & \cdots & \nabla_{\mathbb{P}}m_{2m} \\ \vdots & \vdots & \vdots & \vdots \\ \nabla_{\mathbb{P}}m_{n1} & \nabla_{\mathbb{P}}m_{n2} & \cdots & \nabla_{\mathbb{P}}m_{nm} \end{pmatrix},
    ```
    where ``\mathbb{P}`` are the parameters of the neural network. For computational and consistency reasons each element ``\nabla_\mathbb{P}m_{ij}`` are `NeuralNetworkParameters`.

## Jacobian of a Neural Network

[`SymbolicNeuralNetworks.Jacobian`](@ref) differentiates a symbolic expression with respect to the input arguments of a neural network:

```@example jacobian_gradient
using AbstractNeuralNetworks
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative, params
using Latexify: latexify

c = Chain(Dense(2, 1, tanh; use_bias = false))
nn = SymbolicNeuralNetwork(c)
□ = Jacobian(nn)
# we show the derivative with respect to 
derivative(□) |> latexify
```

Note that the output of `nn` is one-dimensional and we use the convention

```math
\square_{ij} = [\mathrm{jacobian}_{x}f]_{ij} = \frac{\partial}{\partial{}x_j}f_i,
```
so the output has shape ``\mathrm{input\_dim}\times\mathrm{output\_dim} = 1\times2``:

```@example jacobian_gradient
@assert size(derivative(□)) == (1, 2) # hide
size(derivative(□))
```

## Gradient of a Neural Network

As described above [`SymbolicNeuralNetworks.Gradient`](@ref) differentiates every element of the array-valued output with respect to the neural network parameters:

```@example jacobian_gradient
using SymbolicNeuralNetworks: Gradient

g = Gradient(nn)

derivative(g)[1].L1.W |> latexify
```

## Double Derivatives

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
built_function = build_nn_function(derivative(g), params(nn), nn.input)

x = [1., 0.]
ps = NeuralNetworkParameters((L1 = (W = [1. 0.; 0. 1.], b = [0., 0.]), ))
built_function(x, ps)[matrix_index...][layer][weight]
```

!!! info
    With `SymbolicNeuralNetworks`, the `struct`s [`SymbolicNeuralNetworks.Jacobian`](@ref), [`SymbolicNeuralNetworks.Gradient`](@ref) and [`build_nn_function`](@ref) it is easy to build combinations of derivatives. This is much harder when using `Zygote`-based AD.