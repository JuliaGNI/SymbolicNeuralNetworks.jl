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
see [`flatten_equations`](@ref) and [`unflatten`](@ref). Generating one function per entry instead
would re-derive everything the entries have in common — for a symbolic gradient that is the whole
forward pass, once per parameter array — and would compile one `RuntimeGeneratedFunction` per entry
rather than one in total.
"""
function build_nn_function(eqs::EquationSet, sparams::NeuralNetworkParameters,
                           svariables::SymbolicVariables...; kwargs...)
    flat, template = flatten_equations(eqs)
    joint = build_nn_function(flat, sparams, svariables...; kwargs...)
    EquationSetFunction{length(svariables)}(joint, template)
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
function build_nn_function(eqs::AbstractArray{<:EquationSet}, sparams::NeuralNetworkParameters,
                           svariables::SymbolicVariables...; kwargs...)
    functions = map(eq -> build_nn_function(eq, sparams, svariables...; kwargs...), eqs)
    EquationSetArrayFunction{length(svariables)}(functions)
end

"""
    EquationSetFunction{NDATA}(f, template)

The function [`build_nn_function`](@ref) returns for an [`EquationSet`](@ref): it evaluates the
jointly generated `f` and puts the flat result back into the nesting recorded in `template`.
"""
struct EquationSetFunction{NDATA, FT, TT} <: Function
    f::FT
    template::TT
end

EquationSetFunction{NDATA}(f::FT, template::TT) where {NDATA, FT, TT} =
    EquationSetFunction{NDATA, FT, TT}(f, template)

(f::EquationSetFunction{1})(input, ps) = unflatten(f.template, f.f(input, ps))
(f::EquationSetFunction{2})(input, output, ps) = unflatten(f.template, f.f(input, output, ps))

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

@doc raw"""
    FlatSlice(range, size)

Where an entry of an equation set ended up in the flat vector that [`flatten_equations`](@ref)
produces, and which shape it has to be given again by [`unflatten`](@ref). A scalar-valued entry
has `size == ()`.
"""
struct FlatSlice{N}
    range::UnitRange{Int}
    size::NTuple{N, Int}
end

@doc raw"""
    flatten_equations(eqs)

Concatenate every entry of `eqs` into one vector of scalar equations, together with a *template*: a
copy of the nesting of `eqs` in which each entry has been replaced by the [`FlatSlice`](@ref)
describing where it went. [`unflatten`](@ref) reverses this.

# Examples

```jldoctest
using SymbolicNeuralNetworks: flatten_equations, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, params

c = Chain(Dense(2, 3, tanh))
snn = SymbolicNeuralNetwork(c)
flat, template = flatten_equations((a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2))
(length(flat), template.a.range, template.b.range)

# output

(6, 1:3, 4:6)
```
"""
function flatten_equations(eqs::EquationSet)
    flat = Num[]
    template = flatten_equations!(flat, eqs)
    flat, template
end

flatten_equations!(flat::AbstractVector, eqs::NeuralNetworkParameters) =
    NeuralNetworkParameters{keys(eqs)}(map(eq -> flatten_equations!(flat, eq), values(eqs)))
flatten_equations!(flat::AbstractVector, eqs::NamedTuple) =
    NamedTuple{keys(eqs)}(map(eq -> flatten_equations!(flat, eq), values(eqs)))

function flatten_equations!(flat::AbstractVector, eq)
    scalarized = scalar_expressions(eq)
    offset = length(flat)
    append!(flat, _flat_entries(scalarized))
    FlatSlice((offset + 1):length(flat), _equation_size(scalarized))
end

_flat_entries(eq::AbstractArray) = vec(eq)
_flat_entries(eq) = [eq]

@doc raw"""
    unflatten(template, out)

Split the flat result `out` of a jointly generated function back into the nesting recorded in
`template`. The inverse of [`flatten_equations`](@ref).

# Implementation

Each entry is *copied* out of `out` rather than viewed into it, so that the entries are ordinary
`Array`s and cannot alias each other.

`out` is dispatched on by its number of dimensions, which is how the layout of the batch is
accounted for — see [`AbstractBatchedFunction`](@ref) for where those layouts come from:
- a vector when the per-sample results were summed or a single sample was evaluated, in which case
  every entry keeps the shape of its equation,
- a ``P\times{}N`` matrix when they were concatenated, in which case an entry of size
  ``(m, n, \ldots)`` becomes an ``m\times(n\cdot\ldots\cdot{}N)`` matrix,
- a ``P\times{}N_1\times{}N_2`` array when they were concatenated over two batch dimensions.

A scalar-valued entry is treated as one of size ``m = 1`` throughout, so it comes back as a number
for a single sample and as a ``1\times{}N`` matrix for a concatenated batch.
"""
unflatten(template::NeuralNetworkParameters, out::AbstractArray) =
    NeuralNetworkParameters{keys(template)}(map(t -> unflatten(t, out), values(template)))
unflatten(template::NamedTuple, out::AbstractArray) =
    NamedTuple{keys(template)}(map(t -> unflatten(t, out), values(template)))

unflatten(slice::FlatSlice, out::AbstractVector) = _reshape_entry(out[slice.range], slice.size)
unflatten(slice::FlatSlice, out::AbstractMatrix) =
    reshape(out[slice.range, :], _batched_size(slice.size, size(out, 2))...)
function unflatten(slice::FlatSlice, out::AbstractArray{<:Any, 3})
    # the same restriction the single-equation path applies in `_restore_batch_dimensions`: an entry
    # whose result is more than one column wide per sample has no room for a second batch dimension
    trailing_dimensions(slice.size) == 1 || throw(ArgumentError(two_batch_dimension_message(slice.size)))
    out[slice.range, :, :]
end

_reshape_entry(entries::AbstractVector, ::Tuple{}) = entries[begin]
_reshape_entry(entries::AbstractVector, size::Tuple) = reshape(entries, size...)

_batched_size(::Tuple{}, batch_size::Integer) = (1, batch_size)
_batched_size(size::Tuple, batch_size::Integer) = (size[1], prod(Base.tail(size)) * batch_size)
