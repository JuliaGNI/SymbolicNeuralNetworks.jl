"""
    build_nn_function(eqs::AbstractArray{<:NeuralNetworkParameters}, sparams, sinput...)

Build an executable function based on `eqs` that potentially also has a symbolic output.
"""
function build_nn_function(eqs::AbstractArray{<:NeuralNetworkParameters}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...)
    ps = [function_valued_parameters(eq, sprams, sinput...) for eq in eqs]
    ps_semi = [build_nn_function(ps_single, sparams, sinput...) for ps_single in ps]
    pbs_executable(ps, params, input...) = apply_element_wise(ps, params, input...)
    pbs_executable(params, input...) = pbs_executable(ps_semi, params, input...)
    pbs_executable
end

function build_nn_function(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams::NeuralNetworkParameters, sinput...)
    ps = function_valued_parameters(eqs, sparams, sinput...)
    pbs_executable(ps, params, input...) = apply_element_wise(ps, params, input...)
    pbs_executable(params, input...) = pbs_executable(ps, params, input...)
    pbs_executable
end


"""
    function_valued_parameters(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams, sinput...)

Return an executable function for each entry in `eqs`. This still has to be processed with [`apply_element_wise`](@ref).
"""
function function_valued_parameters(eqs::Union{NamedTuple, NeuralNetworkParameters}, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr...)
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
ps = [NeuralNetworkParameters((val1 = (input, params) -> input + params.a, val2 = (input, params) -> input + params.b))]
apply_element_wise(ps, params, 1.)

# output

1-element Vector{NeuralNetworkParameters{(:val1, :val2), Tuple{Float64, Float64}}}:
 NeuralNetworkParameters{(:val1, :val2), Tuple{Float64, Float64}}((val1 = 2.0, val2 = 3.0))
```

Matrix: 

```jldoctest
using SymbolicNeuralNetworks: apply_element_wise
using AbstractNeuralNetworks: NeuralNetworkParameters

# parameter values
params = NeuralNetworkParameters((a = 1., b = 2.))
sc_ps = NeuralNetworkParameters((val1 = (input, params) -> input + params.a, val2 = (input, params) -> input + params.b))
ps = [sc_ps sc_ps]
apply_element_wise(ps, params, 1.) |> typeof

# output

Matrix{NeuralNetworkParameters{(:val1, :val2), Tuple{Float64, Float64}}}
```

# Implementation

This is generating a `@generated function`.
"""
function apply_element_wise(ps::AbstractArray, params::NeuralNetworkParameters, input...)
    apply_element_wise(ps, params, Val(axes(ps)), input...)
end

strip_of_val(::Type{Val{T}}) where T = T

generate_symbols(array_axes::Tuple{Base.OneTo{<:Integer}, Base.OneTo{<:Integer}}) = [gensym() for _ in array_axes[1], __ in array_axes[2]]
generate_symbols(array_axes::Tuple{Base.OneTo{<:Integer}}) = [gensym() for _ in array_axes[1]]

@generated function apply_element_wise(ps::AbstractVector, params::NeuralNetworkParameters, ax::Val, input...)
    array_axes = strip_of_val(ax)
    x_symbols = generate_symbols(array_axes)
    eqs = [:($x_symbol = apply_element_wise(ps[$i], params, input...)) for (x_symbol, i) in zip(x_symbols, array_axes[1])]
    calls = [eqs..., :(return vcat($(x_symbols...)))]
    Expr(:block, calls...)
end

@generated function apply_element_wise(ps::AbstractMatrix, params::NeuralNetworkParameters, ax::Val, input...)
    array_axes = strip_of_val(ax)
    x_symbols = generate_symbols(array_axes)
    eqs = [:($(x_symbols[i, j]) = apply_element_wise(ps[$i, $j], params, input...)) for i ∈ array_axes[1], j ∈ array_axes[2]]
    calls = [eqs..., :(return reshape(vcat($(x_symbols...)), $(array_axes[1].stop), $(array_axes[2].stop)))]
    Expr(:block, calls...)
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
    eqs = [:($x_symbol = ps[$i, $j](input, output, params, input)) for (x_symbol, i) in zip(x_symbols, 1:N)]
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