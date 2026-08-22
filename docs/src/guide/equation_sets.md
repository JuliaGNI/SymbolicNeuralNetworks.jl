```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Equation Sets

[`build_nn_function`](@ref) also accepts a whole *set* of equations: an arbitrarily nested
`NamedTuple` or `NetworkParameters` whose leaves are symbolic expressions. The result is one
function whose output has the same nesting.

```@example sets
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)

eqs = (output = c(snn.input, params(snn)),
       squared = c(snn.input, params(snn)) .^ 2,
       total = sum(c(snn.input, params(snn))))

f = build_nn_function(eqs, params(snn), snn.input)
f([1.0, 2.0], ps)
```

This is not just a convenience. The entries are generated as a **single** function, so everything
they have in common is computed once:

```@example sets
f(rand(2, 4), ps)
```

The entries follow the shape rules of [Building Functions](@ref) individually — a vector-valued entry
becomes a matrix over a batch, a scalar-valued one a ``1\times{}N`` matrix — and each is copied out
of the joint result, so the entries never alias one another.

## Why a single function

The main use of this is a symbolic gradient, whose entries are the derivatives with respect to each
parameter array and therefore all share the entire forward pass:

```@example sets
using SymbolicNeuralNetworks: symbolic_parameter_gradient
using LinearAlgebra: norm

gradient = symbolic_parameter_gradient(norm(c(snn.input, params(snn))) ^ 2, snn)
keys(gradient), keys(gradient.L1)
```

```@example sets
build_nn_function(gradient, params(snn), snn.input; reduce = +)(rand(2, 8), ps).L2.W
```

Building each of those four entries as its own function would re-derive the forward pass four times
and compile four `RuntimeGeneratedFunction`s instead of one. Internally the set is flattened into a
single vector of scalar equations by [`flatten_equations`](@ref) and the flat result is split up
again by [`unflatten`](@ref).

## Arrays of equation sets

An array of equation sets — what [`Gradient`](@ref) produces for an array-valued expression — gives a
function returning an array of results. Each entry of the array is built jointly; the entries
themselves are independent and stay separate functions.

```@example sets
using SymbolicNeuralNetworks: Gradient, derivative

g = build_nn_function(derivative(Gradient(snn)), params(snn), snn.input)
result = g([1.0, 2.0], ps)
length(result), result[1].L1.b
```
