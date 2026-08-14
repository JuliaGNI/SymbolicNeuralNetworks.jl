"""
    build_nn_function(eqs::AbstractArray{<:NeuralNetworkParameters}, sparams, sinput...)

Build an executable function based on an array of symbolic equations `eqs`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: build_nn_function, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

ch = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(ch)
snn = SymbolicNeuralNetwork(nn)
eqs = [(a = ch(snn.input, params(snn)), b = ch(snn.input, params(snn)).^2), (c = ch(snn.input, params(snn)).^3, )]
funcs = build_nn_function(eqs, params(snn), snn.input)
input = [1., 2.]
funcs_evaluated = funcs(input, params(nn))

# output

2-element Vector{NamedTuple}:
 (a = [0.985678060655224], b = [0.9715612392570434])
 (c = [0.9576465981186686],)
```
"""
function build_nn_function(eqs::AbstractArray{<:Union{NamedTuple, NeuralNetworkParameters}}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...; reduce = hcat, cse::Bool = true)
    # every element of `eqs` is generated jointly (see the `NamedTuple` method); the elements
    # themselves are independent, so they stay separate functions
    funcs = [build_nn_function(eq, sparams, sinput...; reduce = reduce, cse = cse) for eq in eqs]

    _pbs_executable(input, params) = _collect_results(funcs, input, params)
    _pbs_executable(input, output, params) = _collect_results(funcs, input, output, params)
    _pbs_executable
end

# `symbolic_pullback` produces a zero-dimensional array when it differentiates a scalar loss.
# As in `apply_element_wise`, that is turned into a one-element vector, which is the shape
# `_get_contents` expects. Arrays of any other dimensionality keep their shape.
_collect_results(funcs::AbstractArray{<:Any, 0}, args...) = [funcs[](args...)]
_collect_results(funcs::AbstractArray, args...) = [f(args...) for f in funcs]

"""
    build_nn_function(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams, sinput...)

Return a function that takes an input, (optionally) an output and neural network parameters and returns a `NeuralNetworkParameters`-valued output.

# Examples

```jldoctest
using SymbolicNeuralNetworks: build_nn_function, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)).^2)
funcs = build_nn_function(eqs, params(snn), snn.input)
input = [1., 2.]
funcs_evaluated = funcs(input, params(nn))

# output

(a = [0.985678060655224], b = [0.9715612392570434])
```

# Implementation

All the entries of `eqs` are generated as a *single* function, whose flat output is split up
again afterwards; see [`flatten_eqs`](@ref) and [`unflatten`](@ref). Generating one function per
entry instead would re-derive everything the entries have in common — for a symbolic pullback
that is the whole forward pass, once per parameter array — and would compile one
`RuntimeGeneratedFunction` per entry rather than one in total.

Equation sets that contain a scalar-valued entry fall back to one function per entry
(via [`function_valued_parameters`](@ref) and [`apply_element_wise`](@ref)), because
`Symbolics.build_function` emits no in-place form for scalars.
"""
function build_nn_function(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...; reduce = hcat, cse::Bool = true)
    flattened = flatten_eqs(eqs)
    isnothing(flattened) && return _build_nn_function_per_leaf(eqs, sparams, sinput...; reduce = reduce, cse = cse)
    flat, template = flattened
    joint = build_nn_function(flat, sparams, sinput...; reduce = reduce, cse = cse)
    _joint_executable(input::AbstractArray, params::NeuralNetworkParameters) = unflatten(template, joint(input, params))
    # return this one if sinput & soutput are supplied
    __joint_executable(input::AbstractArray, output::AbstractArray, params::NeuralNetworkParameters) = unflatten(template, joint(input, output, params))
    typeof(sinput) <: Tuple{<:Any, <:Any} ? __joint_executable : _joint_executable
end

"""
    _build_nn_function_per_leaf(eqs, sparams, sinput...)

Build one function per entry of `eqs`. This is the fallback for equation sets that the joint
code path of [`build_nn_function(::Union{NamedTuple, NeuralNetworkParameters}, ::NeuralNetworkParameters, ::Symbolics.Arr...)`](@ref)
cannot handle.

Internally this is using [`function_valued_parameters`](@ref) and [`apply_element_wise`](@ref).
"""
function _build_nn_function_per_leaf(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...; reduce = hcat, cse::Bool = true)
    ps = function_valued_parameters(eqs, sparams, sinput...; reduce = reduce, cse = cse)
    _pbs_executable(ps::Union{NamedTuple, NeuralNetworkParameters}, params::NeuralNetworkParameters, input::AbstractArray...) = apply_element_wise(ps, params, input...)
    __pbs_executable(input::AbstractArray, params::NeuralNetworkParameters) = _pbs_executable(ps, params, input)
    # return this one if sinput & soutput are supplied
    ___pbs_executable(input::AbstractArray, output::AbstractArray, params::NeuralNetworkParameters) = _pbs_executable(ps, params, input, output)
    typeof(sinput) <: Tuple{<:Any, <:Any} ? ___pbs_executable : __pbs_executable
end

@doc raw"""
    FlatSlice(range, size)

Where an entry of an equation set ended up in the flat vector that [`flatten_eqs`](@ref)
produces, and which shape it has to be given again by [`unflatten`](@ref).
"""
struct FlatSlice{N}
    range::UnitRange{Int}
    size::NTuple{N, Int}
end

@doc raw"""
    flatten_eqs(eqs)

Concatenate every entry of `eqs` into one vector of scalar equations, together with a
*template*: a copy of the nested structure of `eqs` in which each entry has been replaced by
the [`FlatSlice`](@ref) describing where it went. [`unflatten`](@ref) reverses this.

Returns `nothing` if any entry is scalar-valued, which the joint code path cannot handle.

# Examples

```jldoctest
using SymbolicNeuralNetworks: flatten_eqs, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, params

c = Chain(Dense(2, 3, tanh))
snn = SymbolicNeuralNetwork(c)
flat, template = flatten_eqs((a = c(snn.input, params(snn)), b = c(snn.input, params(snn)).^2))
(length(flat), template.a.range, template.b.range)

# output

(6, 1:3, 4:6)
```
"""
function flatten_eqs(eqs::Union{NamedTuple, NeuralNetworkParameters})
    flat = Num[]
    template = _flatten_eqs!(flat, eqs)
    isnothing(template) ? nothing : (flat, template)
end

_flatten_eqs!(flat::AbstractVector, eqs::NeuralNetworkParameters) = _flatten_children!(flat, eqs, NeuralNetworkParameters{keys(eqs)})
_flatten_eqs!(flat::AbstractVector, eqs::NamedTuple) = _flatten_children!(flat, eqs, NamedTuple{keys(eqs)})

function _flatten_children!(flat::AbstractVector, eqs, rewrap)
    children = map(key -> _flatten_eqs!(flat, eqs[key]), keys(eqs))
    any(isnothing, children) && return nothing
    rewrap(children)
end

function _flatten_eqs!(flat::AbstractVector, eq)
    scalarized = Symbolics.scalarize(eq)
    # a scalar entry has no in-place kernel, so the whole set has to take the fallback path
    scalarized isa AbstractArray || return nothing
    offset = length(flat)
    append!(flat, vec(collect(scalarized)))
    FlatSlice((offset + 1):length(flat), size(scalarized))
end

@doc raw"""
    unflatten(template, out)

Split the flat result `out` of a jointly generated function back into the nested structure
recorded in `template`. The inverse of [`flatten_eqs`](@ref).

# Implementation

Each entry is *copied* out of `out` rather than viewed into it, so that the entries are
ordinary `Array`s (as they were when every entry had its own generated function) and cannot
alias each other.

`out` is indexed by its number of dimensions, which is how the layout of the batch is
accounted for:
- a vector when the per-column results were summed (`reduce = +`), in which case every entry
  keeps the shape of its equation,
- a ``P\times{}N`` matrix when they were concatenated (`reduce = hcat`), in which case an entry
  of size ``(m, n, \ldots)`` becomes an ``m\times(n\cdot\ldots\cdot{}N)`` matrix, exactly as
  `Base.reduce(hcat, …)` would have produced.
"""
unflatten(template::NeuralNetworkParameters, out::AbstractArray) = NeuralNetworkParameters{keys(template)}(map(t -> unflatten(t, out), values(template)))
unflatten(template::NamedTuple, out::AbstractArray) = NamedTuple{keys(template)}(map(t -> unflatten(t, out), values(template)))

unflatten(slice::FlatSlice, out::AbstractVector) = reshape(out[slice.range], slice.size...)
function unflatten(slice::FlatSlice, out::AbstractMatrix)
    reshape(out[slice.range, :], slice.size[1], prod(Base.tail(slice.size)) * size(out, 2))
end
unflatten(slice::FlatSlice, out::AbstractArray{<:Any, 3}) = out[slice.range, :, :]

"""
    function_valued_parameters(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams, sinput...)

Return an executable function for each entry in `eqs`. This still has to be processed with [`apply_element_wise`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: function_valued_parameters, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)).^2)
funcs = function_valued_parameters(eqs, params(snn), snn.input)
input = [1., 2.]
ps = params(nn)
a = c(input, ps)
b = c(input, ps).^2

(funcs.a(input, ps), funcs.b(input, ps)) .≈ (a, b)

# output

(true, true)
```
"""
function function_valued_parameters(eqs::NeuralNetworkParameters, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...; reduce = hcat, cse::Bool = true)
    vals = Tuple(build_nn_function(eqs[key], sparams, sinput...; reduce = reduce, cse = cse) for key in keys(eqs))
    NeuralNetworkParameters{keys(eqs)}(vals)
end

function function_valued_parameters(eqs::NamedTuple, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...; reduce = hcat, cse::Bool = true)
    vals = Tuple(build_nn_function(eqs[key], sparams, sinput...; reduce = reduce, cse = cse) for key in keys(eqs))
    NamedTuple{keys(eqs)}(vals)
end

"""
    apply_element_wise(ps, params, input...)

Apply a function element-wise. `ps` is an `Array` where each entry of the array is are `NeuralNetworkParameters` that store functions.
See [`apply_element_wise(::NeuralNetworkParameters, ::NeuralNetworkParameters, ::Any)`](@ref).

# Examples

Vector: 

```jldoctest
using SymbolicNeuralNetworks: apply_element_wise
using AbstractNeuralNetworks: NeuralNetworkParameters

# parameter values
params = NeuralNetworkParameters((a = 1., b = 2.))
ps = [NeuralNetworkParameters((val1 = (input, params) -> input .+ params.a, val2 = (input, params) -> input .+ params.b))]
apply_element_wise(ps, params, [1.])

# output

1-element Vector{NeuralNetworkParameters{(:val1, :val2), Tuple{Vector{Float64}, Vector{Float64}}}}:
 NeuralNetworkParameters{(:val1, :val2), Tuple{Vector{Float64}, Vector{Float64}}}((val1 = [2.0], val2 = [3.0]))
```

Matrix: 

```jldoctest
using SymbolicNeuralNetworks: apply_element_wise
using AbstractNeuralNetworks: NeuralNetworkParameters

# parameter values
params = NeuralNetworkParameters((a = 1., b = 2.))
sc_ps = NeuralNetworkParameters((val1 = (input, params) -> input .+ params.a, val2 = (input, params) -> input .+ params.b))
ps = [sc_ps sc_ps]
apply_element_wise(ps, params, [1.]) |> typeof

# output

Matrix{NeuralNetworkParameters{(:val1, :val2), Tuple{Vector{Float64}, Vector{Float64}}}} (alias for Array{NeuralNetworkParameters{(:val1, :val2), Tuple{Array{Float64, 1}, Array{Float64, 1}}}, 2})
```

# Implementation

This is generating a `@generated function`.
"""
function apply_element_wise(ps::AbstractArray, params::NeuralNetworkParameters, input::AbstractArray...)
    apply_element_wise(ps, params, Val(axes(ps)), input...)
end

strip_of_val(::Type{Val{T}}) where T = T

generate_symbols(array_axes::Tuple{Base.OneTo{<:Integer}, Base.OneTo{<:Integer}}) = [gensym() for _ in array_axes[1], __ in array_axes[2]]
generate_symbols(array_axes::Tuple{Base.OneTo{<:Integer}}) = [gensym() for _ in array_axes[1]]

@generated function apply_element_wise(ps::AbstractVector, params::NeuralNetworkParameters, ax::Val, input::AbstractArray...)
    array_axes = strip_of_val(ax)
    x_symbols = generate_symbols(array_axes)
    eqs = [:($x_symbol = apply_element_wise(ps[$i], params, input...)) for (x_symbol, i) in zip(x_symbols, array_axes[1])]
    calls = [eqs..., :(return vcat($(x_symbols...)))]
    Expr(:block, calls...)
end

@generated function apply_element_wise(ps::AbstractMatrix, params::NeuralNetworkParameters, ax::Val, input::AbstractArray...)
    array_axes = strip_of_val(ax)
    x_symbols = generate_symbols(array_axes)
    eqs = [:($(x_symbols[i, j]) = apply_element_wise(ps[$i, $j], params, input...)) for i ∈ array_axes[1], j ∈ array_axes[2]]
    calls = [eqs..., :(return reshape(vcat($(x_symbols...)), $(array_axes[1].stop), $(array_axes[2].stop)))]
    Expr(:block, calls...)
end

# if the supplied array is of type `Array{<:Any, 0}` then call the vector routine.
function apply_element_wise(ps::AbstractArray{<:Any, 0}, params::NeuralNetworkParameters, ::Val, input::AbstractArray...)
    apply_element_wise([ps[]], params, Val((Base.OneTo(1),)), input...)
end

@generated function apply_element_wise(ps::NamedTuple, params::NeuralNetworkParameters, input)
    N = length(ps.parameters[1])
    x_symbols = [gensym() for _ in 1:N]
    eqs = [:($x_symbol = ps[$i](input, params)) for (x_symbol, i) in zip(x_symbols, 1:N)]
    calls = [eqs..., :(return NamedTuple{$(ps.parameters[1])}(tuple($(x_symbols...))))]
    Expr(:block, calls...)
end

@generated function apply_element_wise(ps::NamedTuple, params::NeuralNetworkParameters, input, output)
    N = length(ps.parameters[1])
    x_symbols = [gensym() for _ in 1:N]
    eqs = [:($x_symbol = ps[$i](input, output, params)) for (x_symbol, i) in zip(x_symbols, 1:N)]
    calls = [eqs..., :(return NamedTuple{$(ps.parameters[1])}(tuple($(x_symbols...))))]
    Expr(:block, calls...)
end

"""
    apply_element_wise(ps, params, input...)

Apply a function element-wise. `ps` is a `NeuralNetworkParameters`-valued function.

# Examples

```jldoctest
using SymbolicNeuralNetworks: apply_element_wise
using AbstractNeuralNetworks: NeuralNetworkParameters

# parameter values
params = NeuralNetworkParameters((a = 1., b = 2.))
ps = NeuralNetworkParameters((val1 = (input, params) -> input + params.a, val2 = (input, params) -> input + params.b))
apply_element_wise(ps, params, 1.)

# output

NeuralNetworkParameters{(:val1, :val2), Tuple{Float64, Float64}}((val1 = 2.0, val2 = 3.0))
```

# Implementation

This is generating a `@generated function`.
"""
@generated function apply_element_wise(ps::NeuralNetworkParameters, params::NeuralNetworkParameters, input)
    N = length(ps.parameters[1])
    x_symbols = [gensym() for _ in 1:N]
    eqs = [:($x_symbol = ps[$i](input, params)) for (x_symbol, i) in zip(x_symbols, 1:N)]
    calls = [eqs..., :(return NeuralNetworkParameters{$(ps.parameters[1])}(tuple($(x_symbols...))))]
    Expr(:block, calls...)
end

@generated function apply_element_wise(ps::NeuralNetworkParameters, params::NeuralNetworkParameters, input, output)
    N = length(ps.parameters[1])
    x_symbols = [gensym() for _ in 1:N]
    eqs = [:($x_symbol = ps[$i](input, output, params)) for (x_symbol, i) in zip(x_symbols, 1:N)]
    calls = [eqs..., :(return NeuralNetworkParameters{$(ps.parameters[1])}(tuple($(x_symbols...))))]
    Expr(:block, calls...)
end