```@meta
CurrentModule = SymbolicNeuralNetworks
```

# SymbolicNeuralNetworks.jl

`SymbolicNeuralNetworks` builds a *symbolic* representation of a (small) neural network, lets you
form arbitrary expressions from it — derivatives with respect to the input, derivatives with respect
to the parameters, and combinations of the two — and compiles those expressions into ordinary `Julia`
functions with [`RuntimeGeneratedFunctions`](https://github.com/SciML/RuntimeGeneratedFunctions.jl).

It is built on [`AbstractNeuralNetworks`](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl) and
is mostly used together with
[`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) and
[`GeometricIntegrators`](https://github.com/JuliaGNI/GeometricIntegrators.jl).

## When to reach for this package

The motivation is [`Zygote`](https://github.com/FluxML/Zygote.jl)'s difficulty with second-order
derivatives: when a loss function itself contains a derivative of the network — as the losses of
Hamiltonian and Lagrangian neural networks do — differentiating it again with reverse-mode AD is
either slow or does not work at all. Computing those derivatives symbolically, once, ahead of time,
side-steps the problem entirely.

The trade-off is that the whole network is unrolled into one expression, so this only makes sense for
*small* networks. Code generation time grows with the size of the network, while evaluation is fast
and allocates little.

## Installation

```julia
using Pkg
Pkg.add("SymbolicNeuralNetworks")
```

## Quickstart

Build a model with `AbstractNeuralNetworks`, wrap it in a [`SymbolicNeuralNetwork`](@ref), and turn
symbolic expressions into functions with [`build_nn_function`](@ref):

```@example quickstart
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 3, tanh), Dense(3, 1, tanh))
snn = SymbolicNeuralNetwork(c)

# the symbolic output of the network, an expression in `snn.input` and `params(snn)`
soutput = c(snn.input, params(snn))
```

```@example quickstart
forward = build_nn_function(soutput, snn)

nn = NeuralNetwork(c)             # the same model, with numeric parameters
forward([1.0, 2.0], params(nn))   # a single sample
```

The generated function also takes a whole batch, one sample per column:

```@example quickstart
forward(rand(2, 4), params(nn))
```

The derivative with respect to the input is a [`Jacobian`](@ref), the derivative with respect to the
parameters a [`Gradient`](@ref), and both are built into functions the same way:

```@example quickstart
using SymbolicNeuralNetworks: Jacobian, derivative

jacobian = build_nn_function(derivative(Jacobian(snn)), snn)
jacobian([1.0, 2.0], params(nn))
```

For training, [`SymbolicPullback`](@ref) provides the derivative of a loss with respect to the
parameters in the shape an optimizer expects:

```@example quickstart
using AbstractNeuralNetworks: FeedForwardLoss

pb = SymbolicPullback(snn, FeedForwardLoss())
loss_value, pullback = pb(params(nn), c, (rand(2, 4), rand(1, 4)))
pullback(1).L1.b
```

## Where to go next

- [Symbolic Neural Networks](@ref) — what a [`SymbolicNeuralNetwork`](@ref) contains and how to build
  expressions from it.
- [Building Functions](@ref) — [`build_nn_function`](@ref) in full: batching, result shapes and the
  `cse`, `inplace` and `reduce` keywords.
- [Derivatives](@ref) — [`Jacobian`](@ref), [`Gradient`](@ref) and [`SymbolicPullback`](@ref).
- [Equation Sets](@ref) — building several equations at once.
- [Flat Parameters](@ref) — functions of a flat parameter vector, and the derivative with respect to
  it, for a solver that wants the network's degrees of freedom as a vector.
- [Training a Symbolic Neural Network](@ref) — a worked example with `GeometricMachineLearning`, and
  how the pullback is built.
- [Limitations](@ref) — the assumptions and rough edges, collected in one place.
- [Code Generation](@ref) — how the generated code is put together, for maintainers.
