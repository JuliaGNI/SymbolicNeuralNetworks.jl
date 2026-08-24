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
    flat, layout = flatten_equations(eqs)
    joint = build_nn_function(flat, sparams, svariables...; kwargs...)
    EquationSetFunction{length(svariables)}(joint, layout)
end

"""
    build_nn_function(eqs::AbstractArray{<:EquationSet}, sparams, svariables...)

Turn an array of equation sets into an executable function that returns an array of results.

Each entry of the array is built by the [`EquationSet`](@ref) method above, i.e. jointly; the
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
    functions = map(eq -> build_nn_function(eq, sparams, svariables...; kwargs...), eqs)
    EquationSetArrayFunction{length(svariables)}(functions)
end

"""
    EquationSetFunction{NDATA}(f, layout)

The function [`build_nn_function`](@ref) returns for an [`EquationSet`](@ref): it evaluates the
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

The function [`build_nn_function`](@ref) returns for an array of [`EquationSet`](@ref)s: it
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
flatten_equations(eqs::EquationSet) = flatten(Num, mapparameters(scalar_expressions, eqs))

@doc raw"""
    split_result(layout, out)

Split the flat result `out` of a jointly generated function into the nesting recorded in `layout`.

`out` is dispatched on by its number of dimensions, which is how the layout of the batch is
accounted for — see [`AbstractBatchedFunction`](@ref) for where those layouts come from. A vector,
which is what a summed batch or a single sample produces, is the case
`NeuralNetworkParameters.unflatten` already covers: every entry simply keeps the shape of its
equation. Anything else has a batch dimension and goes to [`unflatten_batch`](@ref).
"""
split_result(layout::ParameterLayout, out::AbstractVector) = unflatten(layout, out)
split_result(layout::ParameterLayout, out::AbstractArray) = unflatten_batch(layout, out)

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
unflatten_batch(layout::ParametersLayout, out::AbstractArray) =
    NetworkParameters(unflatten_batch(layout.inner, out))
unflatten_batch(layout::NestedLayout, out::AbstractArray) =
    NamedTuple{keys(layout.children)}(map(child -> unflatten_batch(child, out), values(layout.children)))
unflatten_batch(layout::TupleLayout, out::AbstractArray) =
    map(child -> unflatten_batch(child, out), layout.children)
unflatten_batch(layout::WrappedLayout, out::AbstractArray) = unflatten_batch(layout.inner, out)

unflatten_batch(layout::LeafLayout, out::AbstractMatrix) =
    reshape(out[parameterrange(layout), :], _batched_size(layout.size, size(out, 2))...)

function unflatten_batch(layout::LeafLayout, out::AbstractArray{<:Any, 3})
    # the same restriction the single-equation path applies in `_restore_batch_dimensions`: an entry
    # whose result is more than one column wide per sample has no room for a second batch dimension
    trailing_dimensions(layout.size) == 1 || throw(ArgumentError(two_batch_dimension_message(layout.size)))
    out[parameterrange(layout), :, :]
end

_flat_entries(eq::AbstractArray) = vec(eq)
_flat_entries(eq) = [eq]

_batched_size(::Tuple{}, batch_size::Integer) = (1, batch_size)
_batched_size(size::Tuple, batch_size::Integer) = (size[1], prod(Base.tail(size)) * batch_size)
