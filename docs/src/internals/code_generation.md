```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Code Generation

How [`build_nn_function`](@ref) gets from a symbolic expression to a callable object. This page is
for maintainers; nothing here is needed to *use* the package.

## The problem

`Symbolics.build_function(eq, args...; expression = Val{true})` returns an `Expr` for a function that

- takes **one argument per symbolic array** it was given, in that order,
- evaluates the equation for a **single sample**.

What this package needs is a function that

- takes the parameters as **one nested object**, so that it can be called with
  `AbstractNeuralNetworks.params(nn)`,
- evaluates **one column of a batch**, so that a batch costs one allocation rather than one per
  sample.

Bridging that gap is what `src/codegen/` does, in four stages.

## Stage 1: ask Symbolics for the code

[`generated_expression`](@ref) calls `Symbolics.build_function` and picks the out-of-place or the
in-place half of what it returns. The parameters are handed to it as a flat list of arrays by
[`parameter_arguments`](@ref), which also records the access path of each within the parameter
object:

```@example codegen
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: parameter_arguments
using AbstractNeuralNetworks: Chain, Dense, params

c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
snn = SymbolicNeuralNetwork(c)
first(parameter_arguments(params(snn)))
```

Flattening is not optional: `Symbolics.build_function` only recognises a symbolic array that is
passed to it *as an argument*, so handing it the nested parameter object would leave the entries as
free variables in the generated code — which fails at run time with an `UndefVarError`.

For a two-layer network the emitted out-of-place code looks like this:

```julia
function (ˍ₋arg1, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4, ˍ₋arg5)
    @inbounds begin
        var"##cse#1" = (*)(ˍ₋arg1[2], ˍ₋arg2[4])
        var"##cse#2" = (*)(ˍ₋arg1[1], ˍ₋arg2[1])
        ⋮
        (SymbolicUtils.Code.create_array)(typeof(ˍ₋arg1), nothing, Val{1}(), Val{(2,)}(),
                                          var"##cse#17", var"##cse#22")
    end
end
```

## Stage 2: rewrite it

`src/codegen/expression_rewriting.jl` turns that into the body of a kernel, with one small rule per
problem. All of them work on the syntax tree, via [`postwalk`](@ref).

| Rule | What it does |
|------|--------------|
| [`substitute_symbols`](@ref) with [`argument_substitutions`](@ref) | rebinds the generated argument names — *by position* — to `out`, `x1`, `x2` and `ps.L1.W`, `ps.L1.b`, … |
| [`use_generic_array_constructor`](@ref) | `create_array(typeof(…), …)` → `create_array(Array, …)` |
| [`use_base_mapreduce`](@ref) | `Symbolics._mapreduce(…)` → `Base.mapreduce(…; dims = Colon())` |
| [`index_by_batch`](@ref) | `x1[i]` → `x1[i, k]` |
| [`accumulate_into_output`](@ref) | `out[i] = …` → `out[i] += …` or `out[i + (k-1)·L] = …` |

after which the example above reads

```julia
function (x1, ps, k)
    @inbounds begin
        var"##cse#1" = (*)(x1[2, k], ps.L1.W[4])
        var"##cse#2" = (*)(x1[1, k], ps.L1.W[1])
        ⋮
        (SymbolicUtils.Code.create_array)(Array, nothing, Val{1}(), Val{(2,)}(),
                                          var"##cse#17", var"##cse#22")
    end
end
```

A few notes on why each rule is the way it is:

- **Matching arguments by position** is what makes the names irrelevant. It also collapses what used
  to be two nearly identical pipelines — one for a single data argument, one for two — into one:
  the number of data arguments is now just a count.
- **`create_array`** takes the array type to construct as its first argument, and `Symbolics` fills
  that in with the type of one of the arguments. That is a parameter array or a data argument,
  neither of which is the right thing to construct here (`SubArray`, `ReshapedArray`, a `NamedTuple`),
  so it is replaced by the generic `Array`.
- **`index_by_batch`** is why the kernels take a batch index at all. It also *rejects* a data
  argument that is used other than by reading one entry from it, because the batch dimension would
  silently be ignored for that use.
- **`accumulate_into_output`** relies on the in-place code addressing its output with a single
  *linear* index whatever the shape of the equation, which is what makes `(k-1)·L` the offset of
  block `k` in the concatenated result. It insists on exactly one write per entry of the equation:
  a rule that stopped matching would otherwise leave a kernel that still compiles and runs, but
  writes every sample of the batch into the same place.

!!! warning "SymbolicUtils embeds function objects, not symbols"
    In the emitted tree the callee of `(getindex)(x, 1)` is `Base.getindex` **itself**, not the symbol
    `:getindex`; `typeof(…)`, on the other hand, is a symbol. A rule that matches only one of the two
    forms silently does nothing — and in the case of [`index_by_batch`](@ref) the result is still
    *correct for the first sample of a batch*, which makes the bug invisible in a smoke test.
    [`callee_name`](@ref) normalises all of the forms (symbol, function object, `GlobalRef`, qualified
    path) to a plain `Symbol`, and every rule goes through it.

This is also why the rewrites are done on the syntax tree in the first place. They used to be done on
the *printed form* of the code with regexes, which meant a `Meta.parse(string(…))` round-trip to
normalise `(getindex)(x, i)` into `getindex(x, i)`, hard-coded `ˍ₋argN` tokens, and an assertion that
the first argument was literally named `sinput`.

## Stage 3: compile the kernel

[`build_kernel`](@ref) and [`build_kernel!`](@ref) wrap the rewritten body in a function definition
and hand it to `@RuntimeGeneratedFunction`. The resulting signatures are

```julia
kernel(x1, ps, k)               # out-of-place
kernel!(out, x1, ps, k)         # in-place
```

with `x2` inserted after `x1` when there are two data arguments. `build_kernel!` returns `nothing`
for a scalar-valued equation, for which `Symbolics.build_function` emits no in-place form; those take
the out-of-place path.

## Stage 4: add the batching

`src/codegen/batched_function.jl` wraps the kernel in an [`AbstractBatchedFunction`](@ref), which
does everything the kernel does not: iterate over the batch, allocate and shape the result, and
accept a single sample or a three-dimensional batch instead of a matrix. There is one method per rank
of the data arguments, written once for both kernel kinds and both numbers of data arguments.

These are named `struct`s rather than closures, which makes their type concrete and inferable — the
previous implementation returned a local function with several methods, whose return type Julia could
not always infer (`test/codegen/type_stability.jl` guards this).

Equation sets go through one more layer, `src/codegen/equation_sets.jl`: the set is flattened into a
single vector of scalar equations by [`flatten_equations`](@ref), built as one function, and the flat
result is split up again by [`unflatten`](@ref).

## Guarding against upstream drift

Every property of the emitted code that the rules depend on is an implementation detail of
Symbolics/SymbolicUtils and is asserted directly in `test/codegen/codegen_drift.jl`:

- the shape of the emitted function and the order of its arguments,
- that a data argument is only ever read one entry at a time,
- that `create_array` still takes a type as its first argument,
- that the in-place form prepends its output argument and addresses it linearly,
- that a scalar equation still has no in-place form,

each for `cse ∈ (false, true)`, since with CSE the body becomes a `let` block of `var"##cse#N"`
bindings that the rules have to survive too. Most rules also throw by themselves when they stop
matching; the test exists so that drift is reported at its source.
