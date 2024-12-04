# SymbolicNeuralNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/SymbolicNeuralNetworks.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/SymbolicNeuralNetworks.jl/latest/)
[![Build Status](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/SymbolicNeuralNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/SymbolicNeuralNetworks.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SymbolicNeuralNetworks.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SymbolicNeuralNetworks.html)

In a perfect world we probably would not need `SymbolicNeuralNetworks`. Its motivation mainly comes from [`Zygote`](https://github.com/FluxML/Zygote.jl)'s inability to handle second-order derivatives in a decent way[^1]. We also note that if [`Enzyme`](https://github.com/EnzymeAD/Enzyme.jl) matures further, there may be no need for `SymoblicNeuralNetworks` anymore in the future. For now (December 2024) `SymbolicNeuralNetworks` offer a good way to incorporate derivatives into the loss function.

[^1]: In some cases it is possible to perform second-order differentiation with `Zygote`, but when this is possible and when it is not is not entirely clear. 

`SymbolicNeuralNetworks` was created to take advantage of [`Symbolics`](https://symbolics.juliasymbolics.org/stable/) for training neural networks by accelerating their evaluation and by simplifying the computation of arbitrary derivatives of the neural network. This package is based on [`AbstractNeuralNetwork`](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl) and can be applied to [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl). 

`SymbolicNeuralNetworks` creates a symbolic expression of the neural network, computes arbitrary combinations of derivatives and uses [`RuntimeGeneratedFunctions`](https://github.com/SciML/RuntimeGeneratedFunctions.jl) to compile a `Julia` function.

To create a symbolic neural network, we first design a `model` with [`AbstractNeuralNetwork`](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl):
```julia
using AbstractNeuralNetworks

c = Chain(Dense(2, 2, tanh), Linear(2, 1))
```

We now call `SymbolicNeuralNetwork`:

```julia
using SymbolicNeuralNetworks

nn = SymbolicNeuralNetwork(c)
```

## Example

We now train the neural network by using `SymbolicPullback`[^2]:

[^2]: This example is discussed in detail in the docs.

```julia
pb = SymbolicPullback(nn)

using GeometricMachineLearning

# we generate the data and process them with `GeometricMachineLearning.DataLoader`
x_vec = -1.:.1:1.
y_vec = -1.:.1:1.
xy_data = hcat([[x, y] for x in x_vec, y in y_vec]...)
f(x::Vector) = exp.(-sum(x.^2))
z_data = mapreduce(i -> f(xy_data[:, i]), hcat, axes(xy_data, 2))

dl = DataLoader(xy_data, z_data)

nn_cpu = NeuralNetwork(c, CPU())
o = Optimizer(AdamOptimizer(), nn_cpu)
n_epochs = 1000
batch = Batch(10)
o(nn_cpu, dl, batch, n_epochs, pb.loss, pb)
```

We can also train the neural network with `Zygote`-based[^3] automatic differentiation (AD):

[^3]: Note that here we can actually use `Zygote` without problems as it does not involve any complicated derivatives.

```julia
pb_zygote = GeometricMachineLearning.ZygotePullback(FeedForwardLoss())
o(nn_cpu, dl, batch, n_epochs, pb_zygote.loss, pb_zygote)
```

## Development

We are using git hooks, e.g., to enforce that all tests pass before pushing. In order to activate these hooks, the following command must be executed once:
```
git config core.hooksPath .githooks
```