```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Limitations

The assumptions and rough edges of this package, collected in one place.

## Code generation dominates the cost

Everything here is built once and evaluated many times, and the build is the expensive half. Two
things make it so.

The *generated code* re-emits a subexpression once per use, because `Symbolics` stores an expression
as a hash-consed graph but prints it as a tree. Common subexpression elimination — on by default, see
[Building Functions](@ref) — keeps that proportional to the number of distinct operations instead.

The *expression* itself grows with the network when a whole composition is written down as one
formula: a `Chain`'s forward pass inlined layer into layer is exponential in depth. For the pullback
this is avoided by not inlining it, which is what [`SymbolicPullback`](@ref) does by default; see
[How the pullback is built](@ref). Elsewhere — [`Jacobian`](@ref), [`Gradient`](@ref),
[`build_nn_function`](@ref) of an expression over the whole model — the single expression is still
what is built, so those remain practical for small networks only.

`scripts/codegen_comparison.jl` measures both effects.

## The default result cannot be differentiated by `Zygote`

By default a batch is evaluated by an in-place kernel that *mutates* a preallocated array, which
`Zygote` does not support (`Mutating arrays is not supported`). This matters whenever a generated
function is used *inside* a loss that is then differentiated in reverse mode.

Pass `inplace = false` to [`build_nn_function`](@ref) for the out-of-place path, which is
differentiable at the cost of one array per sample. Forward-mode AD (`ForwardDiff`) works with either
path.

## The element type comes from the inputs

The in-place kernels write into an array that has to be allocated *before* they run, so its element
type is promoted over the inputs ([`promoted_eltype`](@ref)) rather than taken from the result.

Two consequences:

- An equation over integer inputs and integer parameters does not generally evaluate to an integer,
  so an integer element type is widened with `float`.
- A `Float32` network whose generated code contains a `Float64` literal produces `Float32`, i.e. the
  literal is rounded rather than the result widened. That is the behaviour one wants for a network,
  but it is worth knowing about.

## `SymbolicPullback` assumes the loss is additive over the batch

[`SymbolicPullback`](@ref) differentiates the loss of a *single* sample and sums the per-sample
gradients over the batch (`reduce = +`). That is the gradient of the batched loss exactly when the
loss is a sum over samples.

`AbstractNeuralNetworks.FeedForwardLoss` is *not*: it normalises by `norm(output)` taken over the
whole batch, so for a batch of more than one sample the symbolic pullback computes the gradient of
the summed per-sample losses, which differs from the gradient of the batched loss. For training this
is a perfectly reasonable objective — and it is what the per-sample formulation means — but it is not
identical to what a `Zygote`-based pullback of the same `NetworkLoss` returns.

The same loss also divides by `norm(output)`, so a target that is identically zero gives `NaN`.

## Batch shapes

- All data arguments of a generated function must have the same number of dimensions and the same
  batch size, however many of them there are.
- A matrix-valued equation cannot be evaluated on a batch with two batch dimensions when
  `reduce = hcat`: concatenating the per-sample results already uses the second dimension. Use
  `reduce = +`, or reshape the input into a matrix.
- `AbstractNeuralNetworks.Dense` computes `ps.W * x`, which has no method for a three-dimensional
  `x`. A generated function handles such an input, but the *model* it came from cannot be called on
  one, which is why references in the test suite are assembled sample by sample. See
  [issue #40](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/40).

## What a layer carries alongside the state

The layerwise construction of [`SymbolicPullback`](@ref) puts a *seam* between two layers, and by
default that seam is one plain vector — so it assumes every layer maps an array to an array. A layer
that carries something else alongside the state can say so ([`seam_interface`](@ref)), within limits:

- the carried data must reach the generated kernels as an array with the *same rank and batch size* as
  the state, like every other data argument. For data that is the same for the whole batch that means
  broadcasting it out to one column per sample.
- a carried datum that varies per sample cannot be combined with a state that has two batch
  dimensions: [`batched`](@ref) lays the state out flat and leaves the carried part alone, so the two
  would no longer correspond.
- the chain's *last* layer has to return the model's output and nothing beside it, since that is what
  the loss and its seed compare against the target.
- a layer that carries nothing declares `()`, not an empty array.
- the carried variables must be named something of their own. `x` is the state and `λ` the
  sensitivities, and reusing either name gives the same symbolic array a second slot rather than a
  second array; the construction refuses such a seam rather than generating kernels for it.

A layer that carries something and declares none of this makes the construction decline, and the
monolithic path takes over. That path traces the chain from a plain vector, so a layer which
*defaults* what it carries — as `GeometricMachineLearning`'s `SymplecticEuler` defaults the parameters
of the system to `NullParameters` — is differentiated with that default, whatever the caller passes.
The gradient is then right for the default-carried map and for no other.

## Reserved names

The generated kernels call their own arguments `out`, `ps`, `k`, and `x1`, `x2`, … — one per data
argument, of which there may be any number.

The whole `x`-followed-by-a-number family is reserved and not merely the arities an equation happens
to use, so that a free variable named `x3` is rejected today rather than the day a third data argument
arrives.

A symbolic variable that is *passed* to [`build_nn_function`](@ref) — as a data variable or as part of
the parameters — may be named anything: `Symbolics.build_function` turns it into an argument and
renames it. Only a variable left **free** in the equation, i.e. one that is neither, survives into the
generated code under its own name. If that name is one of the five above it would be bound by the
kernel's own argument, so it is rejected with an error when the function is built.

## Upstream code generation

The rewrite rules in `src/codegen/expression_rewriting.jl` rely on properties of the code
`Symbolics.build_function` emits which are not part of its documented interface. They are asserted
directly by `test/codegen/codegen_drift.jl`, so a change upstream fails there with a clear message
rather than surfacing as a confusing downstream error. See [Code Generation](@ref).
