"""
    build_nn_function(eqs, nn, soutput)

Build an executable function that can also depend on an output. The resulting `built_function` is then called with:
```julia
built_function(input, output, ps)
```

Also compare this to [`build_nn_function(::EqT, ::AbstractSymbolicNeuralNetwork)`](@ref).

# Extended Help

See the *extended help section* of [`build_nn_function(::EqT, ::AbstractSymbolicNeuralNetwork)`](@ref).
"""
function build_nn_function(eqs, nn::AbstractSymbolicNeuralNetwork, soutput; cse::Bool = true)
    build_nn_function(eqs, params(nn), nn.input, soutput; cse = cse)
end

function build_nn_function(eq::EqT, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr, soutput::Symbolics.Arr; reduce = hcat, cse::Bool = true)
    @assert ( (reduce == hcat) || (reduce == +) ) "Keyword reduce either has to be + or hcat!"
    sc_eq = Symbolics.scalarize(eq)
    kernel! = _build_nn_function_iip(sc_eq, sparams, sinput, soutput; reduce = reduce, cse = cse)
    isnothing(kernel!) && return _oop_batch_wrapper2(_build_nn_function(sc_eq, sparams, sinput, soutput; cse = cse), reduce)
    _iip_batch_wrapper2(kernel!, size(sc_eq), reduce)
end

"""
    _oop_batch_wrapper2(gen_fun, reduce)

The two-input counterpart of [`_oop_batch_wrapper`](@ref).
"""
function _oop_batch_wrapper2(gen_fun, reduce)
    # Single-allocation combine over the batch dimension; see the note in `build_function.jl`
    # on why `mapreduce(…, hcat, …)` is O(N²) and `Base.reduce(hcat, ::Vector)` is O(N).
    gen_fun_returned(input, output, ps) = Base.reduce(reduce, [gen_fun(input, output, ps, k) for k in axes(input, 2)])
    function gen_fun_returned(x::AT, y::AT, ps) where {AT <: Union{AbstractVector, Symbolics.Arr}}
        output_not_reshaped = gen_fun_returned(reshape(x, length(x), 1), reshape(y, length(y), 1), ps)
        # for vectors we do not reshape, as the output may be a matrix
        output_not_reshaped
    end
    # check this! (definitely not correct in all cases!)
    function gen_fun_returned(x::AT, y::AT, ps) where {AT <: AbstractArray{<:Number, 3}}
        output_not_reshaped = gen_fun_returned(reshape(x, size(x, 1), size(x, 2) * size(x, 3)), reshape(y, size(y, 1), size(y, 2) * size(y, 3)), ps)
        # if arrays are added together then don't reshape!
        optional_reshape(output_not_reshaped, reduce, x)
    end
    gen_fun_returned
end

"""
    _iip_batch_wrapper2(kernel!, eq_size, reduce)

The two-input counterpart of [`_iip_batch_wrapper`](@ref).
"""
function _iip_batch_wrapper2(kernel!, eq_size::Tuple, reduce)
    function gen_fun_returned(input, output, ps)
        out = allocate_batch_output(promoted_eltype(input, output, ps), eq_size, size(input, 2), reduce)
        for k in axes(input, 2)
            kernel!(out, input, output, ps, k)
        end
        out
    end
    function gen_fun_returned(x::AT, y::AT, ps) where {AT <: Union{AbstractVector, Symbolics.Arr}}
        output_not_reshaped = gen_fun_returned(reshape(x, length(x), 1), reshape(y, length(y), 1), ps)
        # for vectors we do not reshape, as the output may be a matrix
        output_not_reshaped
    end
    # check this! (definitely not correct in all cases!)
    function gen_fun_returned(x::AT, y::AT, ps) where {AT <: AbstractArray{<:Number, 3}}
        output_not_reshaped = gen_fun_returned(reshape(x, size(x, 1), size(x, 2) * size(x, 3)), reshape(y, size(y, 1), size(y, 2) * size(y, 3)), ps)
        # if arrays are added together then don't reshape!
        optional_reshape(output_not_reshaped, reduce, x)
    end
    gen_fun_returned
end

function optional_reshape(output_not_reshaped::AbstractVecOrMat, ::typeof(+), ::AbstractArray{<:Number, 3})
    output_not_reshaped
end

function optional_reshape(output_not_reshaped::AbstractVecOrMat, ::typeof(hcat), input::AbstractArray{<:Number, 3})
    reshape(output_not_reshaped, size(output_not_reshaped, 1), size(input, 2), size(input, 3))
end

"""
    _build_nn_function(eq, params, sinput, soutput)

Build a function that can process a matrix.
See [`build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref).

# Keyword Arguments

- `cse`: perform *common subexpression elimination* (default `true`). See [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref).

# Implementation

Note that we have two input arguments here which means this method processes code differently than [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr, ::Symbolics.Arr)`](@ref). Here we call:
1. [`fix_create_array`](@ref),
2. [`rewrite_arguments2`](@ref),
3. [`modify_input_arguments2`](@ref),
4. [`fix_map_reduce`](@ref).

See the docstrings for those functions for details on how the code is modified.
"""
function _build_nn_function(eq::EqT, params::NeuralNetworkParameters, sinput::Symbolics.Arr, soutput::Symbolics.Arr; cse::Bool = true)
    sc_eq = Symbolics.scalarize(eq)
    code = build_function_generated(_reduce, sc_eq, sinput, soutput, values(params)...; cse = cse)
    rewritten_code = fix_map_reduce(modify_input_arguments2(rewrite_arguments2(fix_create_array(code))))
    parallelized_code = make_kernel2(rewritten_code)
    @RuntimeGeneratedFunction(parallelized_code)
end

"""
    _build_nn_function_iip(eq, params, sinput, soutput; reduce, cse)

The two-input counterpart of [`_build_nn_function_iip(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref).
The resulting kernel is called with `kernel!(out, input, output, ps, k)`.

# Implementation

As for the single-input case, [`rewrite_arguments2`](@ref) can be reused unchanged: the `ˍ₋out`
argument that `Symbolics.build_function` prepends to the in-place signature is not counted in
the `ˍ₋argN` numbering.
"""
function _build_nn_function_iip(eq::EqT, params::NeuralNetworkParameters, sinput::Symbolics.Arr, soutput::Symbolics.Arr; reduce = hcat, cse::Bool = true)
    sc_eq = Symbolics.scalarize(eq)
    code = build_function_generated(_reduce_iip, sc_eq, sinput, soutput, values(params)...; cse = cse)
    isnothing(code) && return nothing
    rewritten_code = fix_map_reduce(modify_input_arguments_iip2(rewrite_arguments2(fix_create_array(code))))
    parallelized_code = make_kernel_iip2(rewritten_code, reduce, length(sc_eq))
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

"""
    modify_input_arguments_iip2(s)

Change input arguments of type `(ˍ₋out, sinput, soutput, ps.L1, ps.L2)` etc to
`(ˍ₋out, sinput, soutput, ps)`. See [`modify_input_arguments_iip`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: modify_input_arguments_iip2

s = "(ˍ₋out, sinput, soutput, ps.L1, ps.L2, ps.L3)"
modify_input_arguments_iip2(s)

# output
"(ˍ₋out, sinput, soutput, ps)"
```
"""
function modify_input_arguments_iip2(s::AbstractString)
    @assert contains(s, "(ˍ₋out, sinput, soutput, ") "The first input arguments must be ˍ₋out, sinput and soutput."
    regex = r"\(ˍ₋out, sinput, soutput, ps[a-zA-Z0-9., ]+\)"
    replace(s, regex => "(ˍ₋out, sinput, soutput, ps)")
end

function modify_input_arguments_iip2(expression::Expr)
    Meta.parse(modify_input_arguments_iip2(string(expression)))
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

@doc raw"""
    make_kernel_iip2(s, reduce, eq_length)

The two-input counterpart of [`make_kernel_iip`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: make_kernel_iip2

s = "function (ˍ₋out, sinput, soutput, ps)\n begin\n ˍ₋out[1] = getindex(sinput, 1) + getindex(soutput, 2) \n end\n end"
make_kernel_iip2(s, +, 2)

# output

"function (ˍ₋out, sinput, soutput, ps, k)\n begin\n ˍ₋out[1] += getindex(sinput, 1, k) + getindex(soutput, 2, k) \n end\n end"
```
"""
function make_kernel_iip2(s::AbstractString, reduce, eq_length::Integer)
    # add k to function arguments
    s_added_k = replace(s, "function (ˍ₋out, sinput, soutput, ps)" => "function (ˍ₋out, sinput, soutput, ps, k)")
    # add k in body of function
    s_added_k_input = replace(s_added_k, r"getindex\(sinput, ([0-9]+)\)" => s"getindex(sinput, \1, k)")
    s_indexed = replace(s_added_k_input, r"getindex\(soutput, ([0-9]+)\)" => s"getindex(soutput, \1, k)")
    redirect_output_writes(s_indexed, reduce, eq_length)
end

function make_kernel_iip2(expression::Expr, reduce, eq_length::Integer)
    Meta.parse(make_kernel_iip2(string(expression), reduce, eq_length))
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