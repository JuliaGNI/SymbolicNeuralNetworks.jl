"""
    build_nn_function(eqs::EquationSet, sparams, svariables...)

Turn a whole set of equations into one executable function, whose result has the same nesting as
`eqs`.

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
funcs = build_nn_function(eqs, params(snn), snn.input)
funcs([1.0, 2.0], params(nn))

# output

(a = [0.985678060655224], b = [0.9715612392570434])
```

# Implementation

All entries are generated as a *single* function whose flat result is split up again afterwards;
see [`flatten_equations`](@ref) and [`split_result`](@ref). Generating one function per entry
instead would re-derive everything the entries have in common — for a symbolic gradient that is the
whole forward pass, once per parameter array — and would compile one `RuntimeGeneratedFunction` per
entry rather than one in total.
"""
function build_nn_function(eqs::EquationSet, sparams::NetworkParameters,
                           svariables::SymbolicVariables...; kwargs...)
    _build_equation_set_function(eqs, sparams, svariables...; kwargs...)
end

# A symbolic *gradient* is parameter-shaped — it has the shape of the parameters it was taken with
# respect to — so it arrives as a `NetworkParameters` of expressions rather than as an
# [`EquationSet`](@ref). It is a set of equations all the same, and `flatten_equations` has a method
# for each shape, so the two share a body. Written as two methods because they are two questions: one
# takes what a caller wrote, the other what `symbolic_parameter_gradient` returned.
function build_nn_function(eqs::NetworkParameters, sparams::NetworkParameters,
                           svariables::SymbolicVariables...; kwargs...)
    _build_equation_set_function(eqs, sparams, svariables...; kwargs...)
end

function _build_equation_set_function(eqs, sparams, svariables...; kwargs...)
    flat, layout = flatten_equations(eqs)
    joint = build_nn_function(flat, sparams, svariables...; kwargs...)
    EquationSetFunction{length(svariables)}(joint, layout)
end

"""
    build_nn_function(eqs::AbstractArray{<:EquationSet}, sparams, svariables...)

Turn an array of equation sets into an executable function that returns an array of results.

Each entry of the array is built by the `EquationSet` method above, i.e. jointly; the
entries themselves are independent of each other and stay separate functions.

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
eqs = [(a = c(snn.input, params(snn)),), (b = c(snn.input, params(snn)) .^ 3,)]
funcs = build_nn_function(eqs, params(snn), snn.input)
funcs([1.0, 2.0], params(nn))

# output

2-element Vector{NamedTuple{names, Tuple{Vector{Float64}}} where names}:
 (a = [0.985678060655224],)
 (b = [0.9576465981186686],)
```
"""
function build_nn_function(eqs::AbstractArray{<:EquationSet}, sparams::NetworkParameters,
                           svariables::SymbolicVariables...; kwargs...)
    _build_equation_set_array(eqs, sparams, svariables...; kwargs...)
end

# The array counterpart of the gradient shape above: differentiating an array-valued expression gives
# one parameter-shaped set per entry.
function build_nn_function(eqs::AbstractArray{<:NetworkParameters}, sparams::NetworkParameters,
                           svariables::SymbolicVariables...; kwargs...)
    _build_equation_set_array(eqs, sparams, svariables...; kwargs...)
end

function _build_equation_set_array(eqs, sparams, svariables...; kwargs...)
    functions = map(eq -> build_nn_function(eq, sparams, svariables...; kwargs...), eqs)
    EquationSetArrayFunction{length(svariables)}(functions)
end

"""
    EquationSetFunction{NDATA}(f, layout)

The function [`build_nn_function`](@ref) returns for an `EquationSet`: it evaluates the
jointly generated `f` and puts the flat result back into the nesting recorded in `layout`.
"""
struct EquationSetFunction{NDATA, FT, LT} <: Function
    f::FT
    layout::LT
end

EquationSetFunction{NDATA}(f::FT, layout::LT) where {NDATA, FT, LT} =
    EquationSetFunction{NDATA, FT, LT}(f, layout)

(f::EquationSetFunction{1})(input, ps) = split_result(f.layout, f.f(input, ps))
(f::EquationSetFunction{2})(input, output, ps) = split_result(f.layout, f.f(input, output, ps))
# beyond the two common arities the arguments are collected, as for `AbstractBatchedFunction`
(f::EquationSetFunction)(args...) = split_result(f.layout, f.f(args...))

"""
    EquationSetArrayFunction{NDATA}(functions)

The function [`build_nn_function`](@ref) returns for an array of `EquationSet`s: it
evaluates one [`EquationSetFunction`](@ref) per entry and collects the results.
"""
struct EquationSetArrayFunction{NDATA, FT} <: Function
    functions::FT
end

EquationSetArrayFunction{NDATA}(functions::FT) where {NDATA, FT} = EquationSetArrayFunction{NDATA, FT}(functions)

(f::EquationSetArrayFunction{1})(input, ps) = [g(input, ps) for g in f.functions]
(f::EquationSetArrayFunction{2})(input, output, ps) = [g(input, output, ps) for g in f.functions]
(f::EquationSetArrayFunction)(args...) = [g(args...) for g in f.functions]

@doc raw"""
    flatten_equations(eqs)

Concatenate every entry of `eqs` into one vector of scalar equations, together with the
`NeuralNetworkParameters.ParameterLayout` that records where each entry went.

# Examples

```jldoctest
using SymbolicNeuralNetworks: flatten_equations, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, params
using NeuralNetworkParameters: parameterrange

c = Chain(Dense(2, 3, tanh))
snn = SymbolicNeuralNetwork(c)
flat, layout = flatten_equations((a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2))
(length(flat), parameterrange(layout.children.a), parameterrange(layout.children.b))

# output

(6, 1:3, 4:6)
```

# Implementation

The layout is the one `NeuralNetworkParameters` builds for a *parameter* set: an equation set has
the same shape as one, its leaves are arrays (or single instances) of `Num`, and the layout records
exactly what splitting the flat result needs — a range and a size per leaf, in the order the leaves
are written. See [`unflatten_batch`](@ref) for the one thing that has to be added on top, and
`NeuralNetworkParameters.unflatten` for the vector case, which needs nothing.

Each entry is normalised by [`scalar_expressions`](@ref) on the way in, which is what turns a
`Symbolics.Arr` leaf into the `Array{Num}` the layout expects. The element type of the flat vector is
fixed to `Num` rather than left to `NeuralNetworkParameters.parameter_eltype` to promote, so that it
is the same type for every equation set — the code generation downstream dispatches on it.
"""
# Two methods, because two shapes genuinely arrive here. An [`EquationSet`](@ref) is what a caller
# writes; a `NetworkParameters` is what a symbolic *gradient* is, since it has the shape of the
# parameters it was taken with respect to. `flatten` and `mapparameters` handle either, so the body is
# shared rather than the signature widened — the two are different questions and say so.
flatten_equations(eqs::EquationSet) = _flatten_equations(eqs)
flatten_equations(eqs::NetworkParameters) = _flatten_equations(eqs)

_flatten_equations(eqs) = flatten(Num, mapparameters(scalar_expressions, eqs))

@doc raw"""
    split_result(layout, out)

Split the flat result `out` of a jointly generated function into the nesting recorded in `layout`.

`out` is dispatched on by its number of dimensions, which is how the layout of the batch is
accounted for — see [`AbstractBatchedFunction`](@ref) for where those layouts come from. A vector,
which is what a summed batch or a single sample produces, is the case
`NeuralNetworkParameters.unflatten` already covers: every entry simply keeps the shape of its
equation. Anything else has a batch dimension and goes to [`unflatten_batch`](@ref).
"""
@inline split_result(layout::ParameterLayout, out::AbstractVector) = unflatten(layout, out)
@inline split_result(layout::ParameterLayout, out::AbstractArray) = unflatten_batch(layout, out)

@doc raw"""
    unflatten_batch(layout, out)

Split a *batched* flat result into the nesting recorded in `layout`, giving each entry the shape
[`AbstractBatchedFunction`](@ref) documents for a concatenated batch:

- a ``P\times{}N`` matrix, in which an entry of size ``(m, n, \ldots)`` becomes an
  ``m\times(n\cdot\ldots\cdot{}N)`` matrix,
- a ``P\times{}N_1\times{}N_2`` array, in which it becomes an ``m\times{}N_1\times{}N_2`` one.

A scalar-valued entry is treated as one of size ``m = 1`` throughout, so it comes back as a
``1\times{}N`` matrix.

This is deliberately *not* a method of `NeuralNetworkParameters.unflatten`, which already means
something else for a matrix: splitting the rows of a Jacobian taken with respect to a flat parameter
vector, with no batch dimension to restore.

# Implementation

Each entry is *copied* out of `out` rather than viewed into it, so that the entries are ordinary
`Array`s and cannot alias each other.
"""
@inline unflatten_batch(layout::ParametersLayout, out::AbstractArray) =
    NetworkParameters(unflatten_batch(layout.inner, out))
@inline unflatten_batch(layout::NestedLayout, out::AbstractArray) =
    NamedTuple{keys(layout.children)}(_unflatten_batch_children(layout.children, out))
@inline unflatten_batch(layout::TupleLayout, out::AbstractArray) =
    _unflatten_batch_children(layout.children, out)
@inline unflatten_batch(layout::WrappedLayout, out::AbstractArray) = unflatten_batch(layout.inner, out)

# Written out as a `@generated` flat body, in the shape `NeuralNetworkParameters._unflatten_children`
# states as the house rule for walking a layout. Neither of the two obvious alternatives will do, and
# this walk has met both.
#
# `map` over a closure leaves the closure over `out` to be elided, and not every version elides it.
# What that cost was a property of the *shape* of the layout rather than of its depth or its size: on
# Julia 1.10 — the compat floor until 0.7.1 — three leaves in two unequal groups, the shape a `Chain`
# of two `Dense` layers has, cost 640 bytes a call through `map` against 368 without it, while the
# same three leaves laid out flat cost 368 either way. That is issue #55, and it is why this stopped
# being a `map`.
#
# A `Base.tail` chain, which is what it became, fixes that and buys a different problem: `Base.tail`
# yields a new tuple type at every level, so a branch of `k` children costs `k` specialisations over
# argument types each `O(k)` long and inference grows as `k³`. That is `NeuralNetworkParameters`'
# D12, and this walk is exposed to it for the same reason that one was — an equation set is as wide as
# the parameter set it was differentiated from, and a flat set of a few hundred entries is a shape a
# consumer actually has. `NeuralNetworkParameters` 0.2.2 answered it by emitting the body at literal
# indices, and this follows: at 369 children a first call goes from 11.06 s to 1.36 s, for which
# `scripts/batched_walk_cost.jl` is the harness.
#
# One specialisation per branch shape, no closure to elide, no new tuple types, and no `Any32`: `map`
# drops to that fallback past 32 children and it returns a tuple with no concrete type. See
# `test/codegen/allocations.jl`.
@generated function _unflatten_batch_children(layouts, out)
    calls = [:(unflatten_batch(getfield(layouts, $i), out)) for i in 1:fieldcount(layouts)]
    :(($(calls...),))
end

@inline unflatten_batch(layout::LeafLayout, out::AbstractMatrix) =
    reshape(out[parameterrange(layout), :], _batched_size(layout.size, size(out, 2))...)

@inline function unflatten_batch(layout::LeafLayout, out::AbstractArray{<:Any, 3})
    # the same restriction the single-equation path applies in `_restore_batch_dimensions`: an entry
    # whose result is more than one column wide per sample has no room for a second batch dimension
    trailing_dimensions(layout.size) == 1 || throw(ArgumentError(two_batch_dimension_message(layout.size)))
    out[parameterrange(layout), :, :]
end

_flat_entries(eq::AbstractArray) = vec(eq)
_flat_entries(eq) = [eq]

_batched_size(::Tuple{}, batch_size::Integer) = (1, batch_size)
_batched_size(size::Tuple, batch_size::Integer) = (size[1], prod(Base.tail(size)) * batch_size)
