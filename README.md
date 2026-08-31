# SymbolicNeuralNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/SymbolicNeuralNetworks.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/SymbolicNeuralNetworks.jl/latest/)
[![Build Status](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/SymbolicNeuralNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/SymbolicNeuralNetworks.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SymbolicNeuralNetworks.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SymbolicNeuralNetworks.html)

`SymbolicNeuralNetworks` builds a *symbolic* representation of a (small) neural network with
[`Symbolics`](https://symbolics.juliasymbolics.org/stable/), lets you form arbitrary expressions from
it — derivatives with respect to the input, derivatives with respect to the parameters, and
combinations of the two — and compiles those into ordinary `Julia` functions with
[`RuntimeGeneratedFunctions`](https://github.com/SciML/RuntimeGeneratedFunctions.jl). It is built on
[`AbstractNeuralNetworks`](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl) and is meant to be
used together with [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl).

In a perfect world we probably would not need it. Its motivation mainly comes from
[`Zygote`](https://github.com/FluxML/Zygote.jl)'s inability to handle second-order derivatives in a
decent way[^1] — which is exactly what a loss containing a derivative of the network needs. If
[`Enzyme`](https://github.com/EnzymeAD/Enzyme.jl) matures further there may be no need for
`SymbolicNeuralNetworks` in the future; for now it offers a good way to incorporate derivatives into
a loss function.

[^1]: In some cases it is possible to perform second-order differentiation with `Zygote`, but when this is possible and when it is not is not entirely clear.

## Installation

```julia
using Pkg
Pkg.add("SymbolicNeuralNetworks")
```

## Quickstart

Design a `model` with `AbstractNeuralNetworks`, wrap it in a `SymbolicNeuralNetwork`, and compile
symbolic expressions with `build_nn_function`:

```julia
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params

c = Chain(Dense(2, 3, tanh), Dense(3, 1, tanh))
snn = SymbolicNeuralNetwork(c)

forward = build_nn_function(c(snn.input, params(snn)), snn)

nn = NeuralNetwork(c)
forward([1.0, 2.0], params(nn))     # a single sample
forward(rand(2, 8), params(nn))     # a batch, one sample per column
```

Derivatives with respect to the input and with respect to the parameters are built the same way:

```julia
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative

jacobian = build_nn_function(derivative(Jacobian(snn)), snn)
jacobian([1.0, 2.0], params(nn))
```

For training there is `SymbolicPullback`, a drop-in replacement for a `Zygote`-based pullback:

```julia
using AbstractNeuralNetworks: FeedForwardLoss
using GeometricMachineLearning

pb = SymbolicPullback(snn, FeedForwardLoss())

dl = DataLoader(rand(2, 100), rand(1, 100))
nn_cpu = NeuralNetwork(c, CPU())
o = Optimizer(AdamOptimizer(), nn_cpu)
o(nn_cpu, dl, Batch(10), 1000, pb.loss, pb)
```

The pullback is composed layer by layer, so what it costs to build grows with the *size* of the
network rather than exponentially with its depth.

See the [documentation](https://JuliaGNI.github.io/SymbolicNeuralNetworks.jl/latest/) for the full
picture: batching and result shapes, the `cse`/`inplace`/`reduce` keywords, equation sets, functions
of a flat parameter vector, a worked training example, and the limitations of the approach.

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
