"""
    build_nn_function(eq, nn)

Build an executable function based on a symbolic equation, a symbolic input array and a [`SymbolicNeuralNetwork`](@ref).

This function can be called with:

```julia
built_function(input, ps)
```

# Implementation

Internally this is calling [`_build_nn_function`](@ref) and then *parallelizing* the expression via the index `k`.

# Extended Help

The functions mentioned in the implementation section were adjusted ad-hoc to deal with problems that emerged on the fly. 
Other problems may occur. In case you bump into one please [open an issue on github](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues).
"""
function build_nn_function(eq::EqT, nn::AbstractSymbolicNeuralNetwork)
    build_nn_function(eq, nn.params, nn.input)
end

function build_nn_function(eq::EqT, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr; reduce = hcat)
    gen_fun = _build_nn_function(eq, sparams, sinput)
    gen_fun_returned(x, ps) = mapreduce(k -> gen_fun(x, ps, k), reduce, axes(x, 2))
    function gen_fun_returned(x::Union{AbstractVector, Symbolics.Arr}, ps) 
        output_not_reshaped = gen_fun_returned(reshape(x, length(x), 1), ps)
        # for vectors we do not reshape, as the output may be a matrix
        output_not_reshaped
    end
    # check this! (definitely not correct in all cases!)
    function gen_fun_returned(x::AbstractArray{<:Number, 3}, ps) 
        output_not_reshaped = gen_fun_returned(reshape(x, size(x, 1), size(x, 2) * size(x, 3)), ps)
        reshape(output_not_reshaped, size(output_not_reshaped, 1), size(x, 2), size(x, 3))
    end
    gen_fun_returned
end

"""
    _build_nn_function(eq, params, sinput)

Build a function that can process a matrix. This is used as a starting point for [`build_nn_function`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: _build_nn_function, symbolicparameters
using Symbolics
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
params = symbolicparameters(c)
@variables sinput[1:2]
eq = c(sinput, params)
built_function = _build_nn_function(eq, params, sinput)
ps = initialparameters(c)
input = rand(2, 2)

(built_function(input, ps, 1), built_function(input, ps, 2)) .≈ (c(input[:, 1], ps), c(input[:, 2], ps))

# output

(true, true)
```

# Implementation

This first calls `Symbolics.build_function` with the keyword argument `expression = Val{true}` and then modifies the generated code by calling:
1. [`fix_create_array`](@ref),
2. [`rewrite_arguments`](@ref),
3. [`modify_input_arguments`](@ref),
4. [`fix_map_reduce`](@ref).

See the docstrings for those functions for details on how the code is modified. 
"""
function _build_nn_function(eq::EqT, params::NeuralNetworkParameters, sinput::Symbolics.Arr)
    sc_eq = Symbolics.scalarize(eq)
    code = build_function(sc_eq, sinput, values(params)...; expression = Val{true}) |> _reduce_code
    rewritten_code = fix_map_reduce(modify_input_arguments(rewrite_arguments(fix_create_array(code))))
    parallelized_code = make_kernel(rewritten_code)
    @RuntimeGeneratedFunction(parallelized_code)
end

"""
    _reduce_code(code)

Reduce the code.

For some reason `Symbolics.build_function` sometimes returns a tuple and sometimes it doesn't.

This function takes care of this. 
If `build_function` returns a tuple `reduce_code` checks which of the expressions is in-place and then returns the other (not in-place) expression.
"""
function _reduce_code(code::Expr)
    code
end

function _reduce_code(code::Tuple{Expr, Expr})
    contains(string(code[1]), "ˍ₋out") ? code[2] : code[1]
end

"""
    rewrite_arguments(s)

Replace `ˍ₋arg2`, `ˍ₋arg3`, ... with `ps.L1`, `ps.L2` etc.
This is used after `Symbolics.build_function`.

# Examples 

```jldoctest
using SymbolicNeuralNetworks: rewrite_arguments
s = "We test if strings that contain ˍ₋arg2 and ˍ₋arg3 can be converted in the right way."
rewrite_arguments(s)

# output
"We test if strings that contain ps.L1 and ps.L2 can be converted in the right way."
```

# Implementation

The input is first split at the relevant points and then we call [`_modify_integer`](@ref).
The routine [`_modify_integer`](@ref) ensures that we start counting at 1 and not at 2.
By defaut the arguments of the generated function that we get after applying `Symbolics.build_function` are `(x, ˍ₋arg2, ˍ₋arg3)` etc.
We first change this to `(x, ps.L2, ps.L3)` etc. and then to `(x, ps.L1, ps.L2)` etc. via [`_modify_integer`](@ref).
"""
function rewrite_arguments(s::AbstractString)
    regex = r"ˍ₋arg([0-9]+)"
    reformatted = s"ps.L⨸\1⨸"
    expression_with_char = replace(s, regex => reformatted)
    # split at ⨸ symbol:
    expression_split = split(expression_with_char, "⨸")
    *(_modify_integer.(expression_split)...)
end

function rewrite_arguments(expression::Expr)
    Meta.parse(rewrite_arguments(string(expression)))
end

"""
    _modify_integer

If the input is a single integer, subtract 1 from it.

# Examples 

```jldoctest
using SymbolicNeuralNetworks: _modify_integer

s = ["2", "hello", "hello2", "3"]
_modify_integer.(s)

# output
4-element Vector{String}:
 "1"
 "hello"
 "hello2"
 "2"
```
"""
function _modify_integer(s::AbstractString)
    (contains(s, r"[^0-9]+") || isempty(s)) ? s : "$(Meta.parse(s)-1)"
end

"""
    modify_input_arguments(s)

Change input arguments of type `(sinput, ps.L1, ps.L2)` etc to `(sinput, ps)`.
This should be used after [`rewrite_arguments`](@ref). Also see [`build_nn_function`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: modify_input_arguments

s = "(sinput, ps.L1, ps.L2, ps.L3)"
modify_input_arguments(s)

# output
"(sinput, ps)"
```
"""
function modify_input_arguments(s::AbstractString)
    @assert contains(s, "(sinput, ") "The first input argument must be sinput."
    regex = r"\(sinput, ps[a-zA-Z0-9., ]+\)"
    replace(s, regex => "(sinput, ps)")
end

function modify_input_arguments(expression::Expr)
    Meta.parse(modify_input_arguments(string(expression)))
end

"""
   fix_create_array(s)

Fix a problem that occurs in connection with `create_array`.

The function `create_array` from `SymbolicUtils.Code` takes as first input the type of a symbolic array. 
For reasons that are not entirely clear yet the first argument of `create_array` ends up being `ˍ₋arg2`, which is a `NamedTuple` of symoblic arrays.
We solve this problem by replacing `typeof(ˍ₋arg[0-9]+)` with `Array`, which seems to be the most generic possible input to `create_array`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: fix_create_array

s = "(SymbolicUtils.Code.create_array)(typeof(ˍ₋arg2)"
fix_create_array(s)

# output

"SymbolicUtils.Code.create_array(typeof(sinput)"
```

# Implementation

This is used for [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref) and [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr, ::Symbolics.Arr)`](@ref).
"""
function fix_create_array(s::AbstractString)
    @assert contains(s, "ˍ₋arg") "Doesn't contain ˍ₋arg!"
    # replace(s, r"\(SymbolicUtils\.Code\.create_array\)\(typeof\(..arg[0-9]+\), nothing, Val\{1\}\(\), Val\{\(2,\)\}\(\)," => "(")
    replace(s, r"[\(]*SymbolicUtils\.Code\.create_array[\)]*\(typeof\(..arg[0-9]+\)" => "SymbolicUtils.Code.create_array(typeof(sinput)")
end

function fix_create_array(expression::Expr)
    Meta.parse(fix_create_array(string(expression)))
end

"""
    fix_map_reduce(s)

Replace `Symbolics._mapreduce` with `mapreduce` (from `Base`).

When we generate a function with `Symbolics.build_function` it often contains `Symbolics._mapreduce` which cannot be differentiated with Zygote. 
We get around this by replacing `Symbolics._mapreduce` with `mapreduce` and also doing:
```julia
replace(s, ", Colon(), (:init => false,)" => ", dims = Colon()")
```

# Implementation 

This is used for [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref) and [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr, ::Symbolics.Arr)`](@ref).
"""
function fix_map_reduce(s::AbstractString)
    s1 = replace(s, "Symbolics._mapreduce" => "mapreduce")
    replace(s1, ", Colon(), (:init => false,)" => ", dims = Colon()")
end

function fix_map_reduce(expression::Expr)
    Meta.parse(fix_map_reduce(string(expression)))
end

@doc raw"""
# Examples
```jldoctest
using SymbolicNeuralNetworks

s = "function (sinput, ps)\n begin\n getindex(sinput, 1) + getindex(sinput, 2) \n end\n end"
SymbolicNeuralNetworks.make_kernel(s)

# output

"function (sinput, ps, k)\n begin\n getindex(sinput, 1, k) + getindex(sinput, 2, k) \n end\n end"
```
"""
function make_kernel(s::AbstractString)
    # add k to function arguments
    s_added_k = replace(s, "function (sinput, ps)" => "function (sinput, ps, k)")
    # add k in body of function
    replace(s_added_k, r"getindex\(sinput, ([0-9]+)\)" => s"getindex(sinput, \1, k)")
end

function make_kernel(expression::Expr)
    Meta.parse(make_kernel(string(expression)))
end