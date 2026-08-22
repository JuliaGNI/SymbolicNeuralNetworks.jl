```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Double Derivatives

`SymbolicNeuralNetworks` can compute derivatives of arbitrary order, by feeding the symbolic result
of one derivative into the next. This is the case `Zygote`-based AD struggles with, and the reason
this package exists.

The two building blocks are [`Jacobian`](@ref) (differentiate with respect to the input) and
[`Gradient`](@ref) (differentiate with respect to the parameters); see [Derivatives](@ref) for each
on its own.

## Jacobian of a neural network

```@example jacobian_gradient
using AbstractNeuralNetworks
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative
using Latexify: latexify

c = Chain(Dense(2, 1, tanh; use_bias = false))
nn = SymbolicNeuralNetwork(c)
□ = Jacobian(nn)
derivative(□) |> latexify
```

The output of `nn` is one-dimensional and the convention is

```math
\square_{ij} = [\mathrm{jacobian}_{x}f]_{ij} = \frac{\partial}{\partial{}x_j}f_i,
```

so the result has shape ``\mathrm{output\_dim}\times\mathrm{input\_dim} = 1\times2``:

```@example jacobian_gradient
@assert size(derivative(□)) == (1, 2) # hide
size(derivative(□))
```

## Gradient of a neural network

[`Gradient`](@ref) differentiates every element of an array-valued expression with respect to the
network parameters:

```@example jacobian_gradient
g = Gradient(nn)

derivative(g)[1].L1.W |> latexify
```

## Combining the two

Feeding the Jacobian into a [`Gradient`](@ref) differentiates the network twice — first with respect
to its input, then with respect to its parameters:

```@example jacobian_gradient
g = Gradient(derivative(□), nn)
nothing # hide
```

The result is a matrix (of the shape of the Jacobian) of parameter sets. To read the derivative of
the first Jacobian entry with respect to the weight `W` of the first layer:

```@example jacobian_gradient
matrix_index = (1, 1)
layer = :L1
weight = :W
derivative(g)[matrix_index...][layer][weight] |> latexify
```

[`build_nn_function`](@ref) turns the whole thing into an executable function. We evaluate it at

```math
x = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad W = \begin{bmatrix} 1 & 0 \end{bmatrix}
```

```@example jacobian_gradient
using AbstractNeuralNetworks: params
using NeuralNetworkParameters: NetworkParameters

built_function = build_nn_function(derivative(g), params(nn), nn.input)

x = [1.0, 0.0]
ps = NetworkParameters((L1 = (W = [1.0 0.0],),))
built_function(x, ps)[matrix_index...][layer][weight]
```

!!! info
    With [`Jacobian`](@ref), [`Gradient`](@ref) and [`build_nn_function`](@ref), combinations of
    derivatives are just function composition on symbolic expressions. Every entry of the result above
    is generated as part of a *single* function, so the forward pass they share is computed once; see
    [Equation Sets](@ref).
