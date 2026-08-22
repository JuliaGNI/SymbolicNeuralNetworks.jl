```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Symbolic Neural Networks

A [`SymbolicNeuralNetwork`](@ref) pairs a model with symbolic stand-ins for its parameters and its
input. Everything else in this package is built from expressions formed with those two.

```@example snn
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, params

c = Chain(Dense(2, 3, tanh), Dense(3, 1, tanh))
snn = SymbolicNeuralNetwork(c)
```

Any of the following can be used to construct one — a model, an architecture, a single layer, or an
`AbstractNeuralNetworks.NeuralNetwork` whose numeric parameters are only used for their shapes:

```julia
SymbolicNeuralNetwork(c)                       # a Chain
SymbolicNeuralNetwork(Dense(2, 3, tanh))       # a single layer, wrapped in a Chain
SymbolicNeuralNetwork(architecture)            # an Architecture
SymbolicNeuralNetwork(architecture, model)
SymbolicNeuralNetwork(NeuralNetwork(c))
```

## The symbolic input and parameters

The input is a vector of symbolic variables, one per input dimension:

```@example snn
snn.input
```

The parameters have the same nesting as the numeric ones — one entry per layer, each holding the
weight matrix and the bias vector — with every leaf replaced by symbolic variables of the same shape:

```@example snn
params(snn).L1.W
```

They are numbered in the order in which they occur in the parameter set, so `W_1` is the weight
matrix of the first layer, `W_2` its bias, `W_3` the weight matrix of the second layer, and so on.
This is done by [`symbolic_variables`](@ref), which works on any nesting of `NamedTuple`s and
`NetworkParameters`.

!!! info "These are arrays of scalar variables, not symbolic arrays"
    `params(snn).L1.W` is a `Matrix{Num}`, not a `Symbolics.Arr`. The difference matters: `Symbolics`
    cannot differentiate with respect to an entry of a `Symbolics.Arr` without scalarising it first,
    and `Symbolics.build_function` cannot generate code for an expression that still contains one.
    Using scalar variables throughout avoids both problems, at the cost of expressions that are fully
    expanded — which is why this package is meant for *small* networks.

## Building expressions

Applying the model to the symbolic input and parameters gives the symbolic output:

```@example snn
soutput = c(snn.input, params(snn))
```

From there, ordinary `Julia` code builds whatever expression is wanted:

```@example snn
using LinearAlgebra: norm

norm(soutput) ^ 2
```

Reductions such as `sum` and `norm` work directly, because the output is an ordinary array of scalar
expressions.

Expressions may also involve variables of your own — a target output, for instance, which is what a
loss function needs:

```@example snn
using Symbolics

starget = Symbolics.variables(:y, 1:1)
(soutput - starget) .^ 2
```

Any expression built this way can be turned into a function with [`build_nn_function`](@ref); see
[Building Functions](@ref).

## Printing expressions

Symbolic expressions get long quickly. [`Latexify`](https://github.com/korsbo/Latexify.jl) renders
them readably:

```@example snn
using Latexify: latexify

latexify(soutput[1])
```
