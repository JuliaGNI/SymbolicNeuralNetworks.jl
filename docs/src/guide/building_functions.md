```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Building Functions

[`build_nn_function`](@ref) turns a symbolic expression into an executable `Julia` function. It is
the main entry point of this package, and this page is its reference.

```@example build
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)

forward = build_nn_function(c(snn.input, params(snn)), snn)
```

## Calling the result

The generated function takes the data and the parameters:

```julia
built_function(input, ps)
built_function(input, output, ps)     # if the equation also involves a target output
```

Which of the two applies is fixed when the function is built — by whether one or two symbolic
variable arrays were given:

```@example build
using Symbolics

starget = Symbolics.variables(:y, 1:2)
residual = build_nn_function((c(snn.input, params(snn)) - starget) .^ 2, snn, starget)
residual(rand(3), rand(2), ps)
```

The parameters are passed as one object with the same nesting the symbolic parameters had, which is
exactly what `AbstractNeuralNetworks.params` returns.

## Batching and result shapes

The data arguments may be a single sample, a batch, or a batch with two batch dimensions:

```@example build
forward(rand(3), ps)          # a single sample: a vector
```

```@example build
forward(rand(3, 5), ps)       # a batch of five: one sample per column
```

```@example build
size(forward(rand(3, 2, 4), ps))   # two batch dimensions
```

All data arguments must have the same number of dimensions and the same batch size.

For an equation of size ``(m, n, \ldots)`` and a batch of ``N`` samples the result is:

| data arguments | `reduce` | result |
|----------------|----------|--------|
| vectors | either | the shape of the equation |
| ``d\times{}N`` matrices | `hcat` | ``m\times(n\cdot\ldots\cdot{}N)`` |
| ``d\times{}N`` matrices | `+` | the shape of the equation |
| ``d\times{}N_1\times{}N_2`` arrays | `hcat` | ``m\times{}N_1\times{}N_2`` |
| ``d\times{}N_1\times{}N_2`` arrays | `+` | the shape of the equation |

A scalar-valued equation counts as ``m = 1``, so batching it with `hcat` gives a ``1\times{}N``
matrix:

```@example build
total = build_nn_function(sum(c(snn.input, params(snn))), snn)
total(rand(3, 5), ps)
```

A matrix-valued equation batched with `hcat` places the blocks next to each other — the same layout
`reduce(hcat, results)` would give:

```@example build
using SymbolicNeuralNetworks: Jacobian, derivative

jacobian = build_nn_function(derivative(Jacobian(snn)), snn)
size(jacobian(rand(3, 5), ps))     # 2×3 per sample, five samples
```

That layout leaves no room for a second batch dimension, so a matrix-valued equation cannot be
evaluated on a three-dimensional batch with `reduce = hcat`. Use `reduce = +`, or reshape the input
into a matrix.

## Keyword arguments

### `reduce`

How the results of the individual samples of a batch are combined: `hcat` (the default) keeps one
result per sample, `+` sums them.

```@example build
summed = build_nn_function(c(snn.input, params(snn)), snn; reduce = +)
summed(rand(3, 5), ps)
```

Summing is what a gradient over a batch needs — the derivative of a sum of per-sample terms is the
sum of the per-sample derivatives — which is why [`SymbolicPullback`](@ref) uses it.

### `inplace`

Whether a batch is evaluated with an in-place kernel (the default) or an out-of-place one.

With `inplace = true` the result is allocated once and the generated code writes every sample into
it, which costs a single allocation per call. The result is produced by *mutation*, which `Zygote`
does not support:

```julia
Zygote.gradient(p -> sum(forward(input, p)), ps)   # ERROR: Mutating arrays is not supported
```

With `inplace = false` each sample is evaluated on its own and the results are combined with
`Base.reduce`. That allocates an array per sample but is differentiable:

```@example build
import Zygote

differentiable = build_nn_function(c(snn.input, params(snn)), snn; inplace = false)
Zygote.gradient(p -> sum(differentiable(rand(3, 5), p)), ps)[1].L1.b
```

Forward-mode AD works either way, as the array the in-place kernel writes into takes its element type
from the inputs.

A scalar-valued equation always takes the out-of-place path: `Symbolics.build_function` emits no
in-place form for one.

### `cse`

Whether *common subexpression elimination* is performed when the code is generated. It is on by
default and there is rarely a reason to turn it off.

`Symbolics` stores an expression as a hash-consed graph but prints it as a tree, so without CSE every
reuse of a subexpression is emitted again and the generated code grows exponentially with the depth
of the network. With `cse = true` the shared subexpressions become bindings in a `let` block instead.
For the gradient of a two-hidden-layer network the difference is roughly an order of magnitude in
code size; `scripts/codegen_comparison.jl` measures it.

`cse = false` produces fully inlined code, which is occasionally useful when inspecting what was
generated.

## What is returned

The result is a callable `struct`, not a closure, so its type is concrete and can be stored in a
typed field:

```@example build
typeof(forward).name.wrapper
```

```@example build
forward
```

See [`InPlaceBatchedFunction`](@ref) and [`OutOfPlaceBatchedFunction`](@ref).
