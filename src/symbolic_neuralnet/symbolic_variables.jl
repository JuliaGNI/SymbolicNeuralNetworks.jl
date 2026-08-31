"""
    symbolic_variables(x, name)

Build symbolic variables with the shape of `x`, named after `name`.

`x` may be a number, an array, or an arbitrarily nested `NamedTuple`/`NetworkParameters` of
those — i.e. anything that can hold the parameters of a neural network. Every leaf gets its own
name, numbered consecutively: `name_1`, `name_2`, …

# Examples

```jldoctest
using SymbolicNeuralNetworks: symbolic_variables

symbolic_variables((a = 1.0, b = [1, 2]), :X)

# output

(a = X_1, b = Symbolics.Num[X_2₁, X_2₂])
```

```jldoctest
using SymbolicNeuralNetworks: symbolic_variables
using AbstractNeuralNetworks: NeuralNetwork, Chain, Dense, params
using NeuralNetworkParameters: NetworkParameters

nn = NeuralNetwork(Chain(Dense(1, 2; use_bias = false), Dense(2, 1; use_bias = false)))
sparams = symbolic_variables(params(nn), :W)
(sparams isa NetworkParameters, keys(sparams), size(sparams.L1.W), eltype(sparams.L1.W))

# output

(true, (:L1, :L2), (2, 1), Symbolics.Num)
```

# Implementation

The variables are *scalar* ones (`Symbolics.variable`/`Symbolics.variables`), so an array of
parameters becomes an `Array{Num}` rather than a `Symbolics.Arr`. `Symbolics` cannot differentiate
with respect to the entries of a `Symbolics.Arr` without scalarising it first, and
`Symbolics.build_function` cannot generate code for expressions that still contain one; using
scalar variables throughout avoids both problems. See [`scalar_expressions`](@ref).
"""
symbolic_variables(x, name::Symbol) = symbolic_variables!(Dict{Symbol, Int}(), x, name)

"""
    symbolic_variables!(counters, x, name)

The workhorse of [`symbolic_variables`](@ref). `counters` maps a name to the number of variables
that have been created under it so far and is updated in place, which is what makes the leaves of a
nested parameter set distinguishable.
"""
function symbolic_variables!(counters::Dict{Symbol, Int}, ::Real, name::Symbol)
    Symbolics.variable(next_name!(counters, name))
end

function symbolic_variables!(counters::Dict{Symbol, Int}, x::AbstractArray, name::Symbol)
    Symbolics.variables(next_name!(counters, name), axes(x)...)
end

function symbolic_variables!(counters::Dict{Symbol, Int}, x::NamedTuple, name::Symbol)
    NamedTuple{keys(x)}(map(value -> symbolic_variables!(counters, value, name), values(x)))
end

function symbolic_variables!(counters::Dict{Symbol, Int}, x::NetworkParameters, name::Symbol)
    NetworkParameters(symbolic_variables!(counters, params(x), name))
end

"""
    next_name!(counters, name)

Return the next unused name derived from `name` and count it in `counters`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: next_name!

counters = Dict{Symbol, Int}()
(next_name!(counters, :var), next_name!(counters, :var))

# output

(:var_1, :var_2)
```
"""
function next_name!(counters::Dict{Symbol, Int}, name::Symbol)
    count = get(counters, name, 0) + 1
    counters[name] = count
    Symbol(name, :_, count)
end
