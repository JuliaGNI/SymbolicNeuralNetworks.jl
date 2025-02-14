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
 (a = [-0.9999386280616135], b = [0.9998772598897417])
 (c = [-0.9998158954841537],)
```
"""
function build_nn_function(eqs::AbstractArray{<:Union{NamedTuple, NeuralNetworkParameters}}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...)
    ps_semi = [function_valued_parameters(eq, sparams, sinput...) for eq in eqs]
    
    _pbs_executable(ps_functions, params, input...) = apply_element_wise(ps_functions, params, input...)
    __pbs_executable(input, params) = _pbs_executable(ps_semi, params, input)
    __pbs_executable(input, output, params) = _pbs_executable(ps_semi, params, input, output)
    __pbs_executable
end

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

(a = [-0.9999386280616135], b = [0.9998772598897417])
```

# Implementation

Internally this is using [`function_valued_parameters`](@ref) and [`apply_element_wise`](@ref).
"""
function build_nn_function(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...)
    ps = function_valued_parameters(eqs, sparams, sinput...)
    _pbs_executable(ps::Union{NamedTuple, NeuralNetworkParameters}, params::NeuralNetworkParameters, input::AbstractArray...) = apply_element_wise(ps, params, input...)
    __pbs_executable(input::AbstractArray, params::NeuralNetworkParameters) = _pbs_executable(ps, params, input)
    # return this one if sinput & soutput are supplied
    ___pbs_executable(input::AbstractArray, output::AbstractArray, params::NeuralNetworkParameters) = _pbs_executable(ps, params, input, output)
    typeof(sinput) <: Tuple{<:Any, <:Any} ? ___pbs_executable : __pbs_executable
end

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
function function_valued_parameters(eqs::NeuralNetworkParameters, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...)
    vals = Tuple(build_nn_function(eqs[key], sparams, sinput...) for key in keys(eqs))
    NeuralNetworkParameters{keys(eqs)}(vals)
end

function function_valued_parameters(eqs::NamedTuple, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...)
    vals = Tuple(build_nn_function(eqs[key], sparams, sinput...) for key in keys(eqs))
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