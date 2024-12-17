"""
    build_nn_function(eqs, nn, soutput)

Build an executable function that can also depend on an output. It is then called with:
```julia
built_function(input, output, ps)
```

Also compare this to [`build_nn_function(::EqT, ::AbstractSymbolicNeuralNetwork)`](@ref).

# Extended Help

See the *extended help section* of [`build_nn_function(::EqT, ::AbstractSymbolicNeuralNetwork)`](@ref).
"""
function build_nn_function(eqs, nn::AbstractSymbolicNeuralNetwork, soutput)
    build_nn_function(eqs, params(nn), nn.input, soutput)
end

function build_nn_function(eq::EqT, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr, soutput::Symbolics.Arr)
    gen_fun = _build_nn_function(eq, sparams, sinput, soutput)
    gen_fun_returned(input, output, ps) = mapreduce(k -> gen_fun(input, output, ps, k), +, axes(input, 2))
    gen_fun_returned(input::AT, output::AT, ps) where {AT <: Union{AbstractVector, Symbolics.Arr}} = gen_fun_returned(reshape(input, length(input), 1), reshape(output, length(output), 1), ps)
    gen_fun_returned(input::AT, output::AT, ps) where {T, AT <: AbstractArray{T, 3}} = gen_fun_returned(reshape(input, size(input, 1), size(input, 2) * size(input, 3)), reshape(output, size(output, 1), size(output, 2) * size(output, 3)), ps)
    gen_fun_returned
end

"""
    _build_nn_function(eq, params, sinput, soutput)

Build a function that can process a matrix.
See [`build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref).

# Implementation

Note that we have two input arguments here which means this method processes code differently than [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr, ::Symbolics.Arr)`](@ref). Here we call:
1. [`fix_create_array`](@ref),
2. [`rewrite_arguments2`](@ref),
3. [`modify_input_arguments2`](@ref),
4. [`fix_map_reduce`](@ref).

See the docstrings for those functions for details on how the code is modified. 
"""
function _build_nn_function(eq::EqT, params::NeuralNetworkParameters, sinput::Symbolics.Arr, soutput::Symbolics.Arr)
    sc_eq = Symbolics.scalarize(eq)
    code = build_function(sc_eq, sinput, soutput, values(params)...; expression = Val{true}) |> _reduce_code
    rewritten_code = fix_map_reduce(modify_input_arguments2(rewrite_arguments2(fix_create_array(code))))
    parallelized_code = make_kernel2(rewritten_code)
    @RuntimeGeneratedFunction(parallelized_code)
end

"""
    modify_input_arguments2(s)

Change input arguments of type `(sinput, soutput, ps.L1, ps.L2)` etc to `(sinput, soutput, ps)`.
This should be used after [`rewrite_arguments`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: modify_input_arguments2

s = "(sinput, soutput, ps.L1, ps.L2, ps.L3)"
modify_input_arguments2(s)

# output
"(sinput, soutput, ps)"
```
"""
function modify_input_arguments2(s::AbstractString)
    @assert contains(s, "(sinput, soutput, ") "The first input arguments must be sinput and soutput."
    regex = r"\(sinput, soutput, ps[a-zA-Z0-9., ]+\)"
    replace(s, regex => "(sinput, soutput, ps)")
end

function modify_input_arguments2(expression::Expr)
    Meta.parse(modify_input_arguments2(string(expression)))
end

@doc raw"""
# Examples
```jldoctest
using SymbolicNeuralNetworks

s = "function (sinput, soutput, ps)\n begin\n getindex(sinput, 1) + getindex(soutput, 2) \n end\n end"
SymbolicNeuralNetworks.make_kernel2(s)

# output

"function (sinput, soutput, ps, k)\n begin\n getindex(sinput, 1, k) + getindex(soutput, 2, k) \n end\n end"
```
"""
function make_kernel2(s::AbstractString)
    # add k to function arguments
    s_added_k = replace(s, "function (sinput, soutput, ps)" => "function (sinput, soutput, ps, k)")
    # add k in body of function
    s_added_k_input = replace(s_added_k, r"getindex\(sinput, ([0-9]+)\)" => s"getindex(sinput, \1, k)")
    replace(s_added_k_input, r"getindex\(soutput, ([0-9]+)\)" => s"getindex(soutput, \1, k)")
end

function make_kernel2(expression::Expr)
    Meta.parse(make_kernel2(string(expression)))
end

"""
    rewrite_arguments2(s)

Replace `ˍ₋arg3`, `ˍ₋arg4`, ... with `ps.L1`, `ps.L2` etc.
Note that we subtract two from the input, unlike [`rewrite_arguments`](@ref) where it is one.

# Examples 

```jldoctest
using SymbolicNeuralNetworks: rewrite_arguments2
s = "We test if strings that contain ˍ₋arg3 and ˍ₋arg4 can be converted in the right way."
rewrite_arguments2(s)

# output
"We test if strings that contain ps.L1 and ps.L2 can be converted in the right way."
```

# Implementation

The input is first split at the relevant points and then we call [`_modify_integer2`](@ref).
The routine [`_modify_integer2`](@ref) ensures that we start counting at 1 and not at 3.
See [`rewrite_arguments`](@ref).
"""
function rewrite_arguments2(s::AbstractString)
    regex = r"ˍ₋arg([0-9]+)"
    reformatted = s"ps.L⨸\1⨸"
    expression_with_char = replace(s, regex => reformatted)
    # split at ⨸ symbol:
    expression_split = split(expression_with_char, "⨸")
    *(_modify_integer2.(expression_split)...)
end

function rewrite_arguments2(expression::Expr)
    Meta.parse(rewrite_arguments2(string(expression)))
end

"""
    _modify_integer2

If the input is a single integer, subtract 2 from it.

# Examples 

```jldoctest
using SymbolicNeuralNetworks: _modify_integer2

s = ["3", "hello", "hello2", "4"]
_modify_integer2.(s)

# output
4-element Vector{String}:
 "1"
 "hello"
 "hello2"
 "2"
```
"""
function _modify_integer2(s::AbstractString)
    (contains(s, r"[^0-9]+") || isempty(s)) ? s : "$(Meta.parse(s)-2)"
end