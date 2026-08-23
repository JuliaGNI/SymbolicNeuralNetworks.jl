```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Flat Parameters

A solver does not want a parameter set. A Newton iteration, a least-squares fit or a quasi-Newton
method wants the unknowns as one vector, and the derivative with respect to that vector as a matrix.

The conversion between the two shapes is not this package's — it lives in
[`NeuralNetworkParameters`](https://github.com/JuliaGNI/NeuralNetworkParameters.jl), which owns the
parameter container itself. `flatten` copies every number of a parameter set into one vector and hands
back the `ParameterLayout` that puts it together again:

```@example flat
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: flatten, unflatten, flatlength
import Random
Random.seed!(123)

c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
nn = NeuralNetwork(c)
ps = params(nn)

w, layout = flatten(ps)
(flatlength(ps), unflatten(layout, w) == ps)
```

A layout is a *value*: built once, then stored in a solver's cache and reused. There are
allocation-free `flatten!`/`unflatten!` variants for inner loops, `ChainRulesCore` rules so that
reverse mode goes through the conversion, and a `FlatParameters` wrapper that is an `AbstractVector`
carrying its own layout. See that package's documentation for all of it.

What this package adds is the symbolic half: generating a function that *takes* the flat form, and the
derivative with respect to it.

## Functions of a flat parameter vector

[`build_flat_function`](@ref) is [`build_nn_function`](@ref) with a flat parameter argument:

```@example flat
using SymbolicNeuralNetworks: build_flat_function

snn = SymbolicNeuralNetwork(c)
f = build_flat_function(c(snn.input, params(snn)), snn)

f([1.0, 2.0], w) ≈ c([1.0, 2.0], ps)
```

It takes the same keyword arguments as [`build_nn_function`](@ref) and accepts everything it accepts,
including an [equation set](@ref "Equation Sets"). A `FlatParameters` works too, and is read through
the layout it carries rather than the one the function was built with.

Because the vector is laid out out-of-place, it may have a different element type from the parameters
the layout was built from — which is what makes the flat form usable for derivatives:

```@example flat
import ForwardDiff

input = rand(2)
ForwardDiff.jacobian(v -> f(input, v), w) |> size
```

## The derivative with respect to the flat parameters

Asking `ForwardDiff` for that Jacobian works, but the point of this package is to have it
symbolically. [`flat_parameter_gradient`](@ref) differentiates with respect to the parameters and
lays the result out flat — a vector for a scalar expression, and for an array-valued one the
``\mathrm{length}(f)\times\mathrm{flatlength}`` Jacobian with rows indexed by `vec(f)`:

```@example flat
using SymbolicNeuralNetworks: flat_parameter_gradient

J = flat_parameter_gradient(c(snn.input, params(snn)), snn)
size(J)
```

```@example flat
jacobian = build_nn_function(J, snn)
jacobian(input, ps) ≈ ForwardDiff.jacobian(v -> f(input, v), w)
```

A column block of that matrix belongs to one entry of the parameter set, and `unflatten`'s matrix
method reads it back — the same layout, used for the other of its two meanings:

```@example flat
blocks = unflatten(layout, permutedims(jacobian(input, ps)))
size(blocks.L1.W)          # the 6 entries of L1.W against the 2 outputs
```

## A nonlinear solve

Put the two together and both directions are flat, which is the shape a Newton step is assembled
from — the residual of a network evaluated against a target, and its derivative with respect to the
network's degrees of freedom:

```@example flat
using Symbolics

soutput = Symbolics.variables(:y, 1:2)
equation = c(snn.input, params(snn)) - soutput

residual = build_flat_function(equation, snn, soutput)
jacobian = build_flat_function(flat_parameter_gradient(equation, snn), snn, soutput)

x, y = rand(2), rand(2)
(residual(x, y, w), size(jacobian(x, y, w)))
```

The same pattern works for degrees of freedom that are not the parameters of a network at all: the
layout machinery upstream is written against an arbitrarily nested collection of arrays, not against
neural networks, so anything shaped like one flattens.
