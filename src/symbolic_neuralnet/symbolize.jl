"""
    symboliccounter!(cache, arg; redundancy)

Add a specific argument to the cache.

# Examples

```jldoctest
using SymbolicNeuralNetworks: symboliccounter!

cache = Dict()
var = symboliccounter!(cache, :var)
(cache, var)

# output
(Dict{Any, Any}(:var => 1), :var_1)

```
"""
function symboliccounter!(cache::Dict, arg::Symbol; redundancy::Bool = true)
    if redundancy
        arg ∈ keys(cache) ? cache[arg] += 1 : cache[arg] = 1
        nam = string(arg) * "_" * string(cache[arg])
        Symbol(nam)
    else
        arg
    end
end

"""
    symbolize!(cache, nt, var_name)

Symbolize all the arguments in `nt`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: symbolize!

cache = Dict()
sym = symbolize!(cache, .1, :X)
(sym, cache)

# output

(X_1, Dict{Any, Any}(:X => 1))
```

```jldoctest
using SymbolicNeuralNetworks: symbolize!

cache = Dict()
arr = rand(2, 1)
sym_scalar = symbolize!(cache, .1, :X)
sym_array = symbolize!(cache, arr, :Y)
(sym_array, cache)

# output

(Y_1[Base.OneTo(2),Base.OneTo(1)], Dict{Any, Any}(:X => 1, :Y => 1))
```

Note that the for the second case the cache is storing a scalar under `:X` and an array under `:Y`. If we use the same label for both we get:

```jldoctest
using SymbolicNeuralNetworks: symbolize!

cache = Dict()
arr = rand(2, 1)
sym_scalar = symbolize!(cache, .1, :X)
sym_array = symbolize!(cache, arr, :X)
(sym_array, cache)

# output

(X_2[Base.OneTo(2),Base.OneTo(1)], Dict{Any, Any}(:X => 2))
```

We can also use `symbolize!` with `NamedTuple`s:

```jldoctest
using SymbolicNeuralNetworks: symbolize!

cache = Dict()
nt = (a = 1, b = [1, 2])
sym = symbolize!(cache, nt, :X)
(sym, cache)

# output

((a = X_1, b = X_2[Base.OneTo(2)]), Dict{Any, Any}(:X => 2))
```

And for neural network parameters:

```jldoctest
using SymbolicNeuralNetworks: symbolize!
using AbstractNeuralNetworks: NeuralNetwork, params, Chain, Dense

nn = NeuralNetwork(Chain(Dense(1, 2; use_bias = false), Dense(2, 1; use_bias = false)))
cache = Dict()
sym = symbolize!(cache, params(nn), :X) |> typeof

# output

AbstractNeuralNetworks.NeuralNetworkParameters{(:L1, :L2), Tuple{@NamedTuple{W::Symbolics.Arr{Symbolics.Num, 2}}, @NamedTuple{W::Symbolics.Arr{Symbolics.Num, 2}}}}
```

# Implementation

Internally this is using [`symboliccounter!`](@ref). This function is also adjusting/altering the `cache` (that is optionally supplied as an input argument).
"""
symbolize!

function symbolize!(cache::Dict, ::Real, var_name::Symbol; redundancy::Bool = true)::Symbolics.Num
    sname = symboliccounter!(cache, var_name; redundancy = redundancy)
    (@variables $sname)[1]
end

function symbolize!(cache::Dict, M::AbstractArray, var_name::Symbol; redundancy::Bool = true)
    sname = symboliccounter!(cache, var_name; redundancy = redundancy)
    (@variables $sname[axes(M)...])[1]
end

function symbolize!(cache::Dict, nt::NamedTuple, var_name::Symbol; redundancy::Bool = true)
    values = Tuple(symbolize!(cache, nt[key], var_name; redundancy = redundancy) for key in keys(nt))
    NamedTuple{keys(nt)}(values)
end

function symbolize!(cache::Dict, nt::NeuralNetworkParameters, var_name::Symbol; redundancy::Bool = true)
    NeuralNetworkParameters(symbolize!(cache, params(nt), var_name; redundancy = redundancy))
end