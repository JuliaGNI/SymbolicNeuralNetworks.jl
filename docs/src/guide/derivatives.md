```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Derivatives

There are two directions in which a neural network can be differentiated, and this package has a
`struct` for each:

- [`Jacobian`](@ref) differentiates with respect to the *input*,
- [`Gradient`](@ref) differentiates with respect to the *parameters*.

Both store a symbolic expression, which [`derivative`](@ref) returns and
[`build_nn_function`](@ref) compiles. Because the result of one is again a symbolic expression, they
compose freely — see [Double Derivatives](@ref).

```@example derivatives
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
nothing # hide
```

## Jacobian

[`Jacobian`](@ref) differentiates an expression with respect to `nn.input`. Without an expression it
takes the output of the network:

```@example derivatives
j = Jacobian(snn)
size(derivative(j))
```

The convention is ``\square_{ij} = \partial{}f_i/\partial{}x_j``, so the result is
``\mathrm{output\_dim}\times\mathrm{input\_dim}`` — the same convention `Zygote` and `ForwardDiff`
use:

```@example derivatives
import ForwardDiff

input = rand(2)
jacobian = build_nn_function(derivative(j), snn)
jacobian(input, ps) ≈ ForwardDiff.jacobian(x -> c(x, ps), input)
```

An expression can also be given explicitly, in which case it is flattened with `vec` first, so the
rows of the result are indexed by `vec(f)`:

```@example derivatives
Jacobian(c(snn.input, params(snn)) .^ 2, snn) |> derivative |> size
```

## Gradient

[`Gradient`](@ref) differentiates with respect to `params(snn)`. Its result has the *shape of the
parameters*: for each entry of the differentiated expression there is one full parameter set holding
the derivative with respect to each parameter.

```@example derivatives
g = Gradient(snn)
derivative(g)[1].L1.b
```

!!! info "Terminology"
    The name `Gradient` is not used in the usual sense here. A gradient normally collects the partial
    derivatives of a *scalar* function; [`Gradient`](@ref) differentiates every entry of an array
    with respect to every parameter, so the gradient of a matrix is a matrix of parameter sets:
    ```math
    \mathtt{Gradient}\left( \begin{pmatrix} m_{11} & \cdots & m_{1m} \\ \vdots & \vdots & \vdots \\ m_{n1} & \cdots & m_{nm} \end{pmatrix} \right) = \begin{pmatrix} \nabla_{\mathbb{P}}m_{11} & \cdots & \nabla_{\mathbb{P}}m_{1m} \\ \vdots & \vdots & \vdots \\ \nabla_{\mathbb{P}}m_{n1} & \cdots & \nabla_{\mathbb{P}}m_{nm} \end{pmatrix},
    ```
    where ``\mathbb{P}`` are the parameters of the network.

The underlying function is [`symbolic_parameter_gradient`](@ref), which can also be used directly.
For a *scalar* expression it returns a single parameter set rather than an array of them:

```@example derivatives
using SymbolicNeuralNetworks: symbolic_parameter_gradient
using LinearAlgebra: norm

gradient = symbolic_parameter_gradient(norm(c(snn.input, params(snn))) ^ 2, snn)
keys(gradient)
```

Such a parameter-shaped expression is an [equation set](@ref "Equation Sets"), which
[`build_nn_function`](@ref) builds as a single function whose result has the same nesting. Summing
over the batch with `reduce = +` gives the gradient of the summed per-sample expression:

```@example derivatives
f = build_nn_function(gradient, params(snn), snn.input; reduce = +)
f(rand(2, 8), ps).L1.b
```

## SymbolicPullback

[`SymbolicPullback`](@ref) packages that up for training: it differentiates an
`AbstractNeuralNetworks.NetworkLoss` with respect to the parameters and presents the result the way
an optimizer expects a pullback to look.

```@example derivatives
using AbstractNeuralNetworks: FeedForwardLoss

pb = SymbolicPullback(snn, FeedForwardLoss())

input, output = rand(2, 8), rand(2, 8)
loss_value, pullback = pb(params(nn), c, (input, output))
loss_value
```

The second entry is a function of the *output sensitivities*. A `NetworkLoss` is scalar-valued, so
those are just `1`:

```@example derivatives
pullback(1).L1.b
```

Constructing the pullback is where most of the time goes; evaluating it is fast and allocates once
per call. See [Training a Symbolic Neural Network](@ref) for a complete training run, and
[Limitations](@ref) for the assumption `SymbolicPullback` makes about the loss.
