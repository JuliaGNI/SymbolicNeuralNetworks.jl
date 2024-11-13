"""
    build_nn_function(eq, sinput, nn)

Build an executable function based on a symbolic equation, a symbolic input array and a [`SymbolicNeuralNetwork`](@ref).

# Implementation

This first calls `Symbolics.build_function` with the keyword argument `expression = Val{true}` and then modifies the generated code by calling:
1. [`fix_create_array`](@ref),
2. [`rewrite_arguments`](@ref),
3. [`modify_input_arguments`](@ref),
4. [`fix_map_reduce`](@ref).

See the docstrings for those four functions for details on how the code is modified. 

# Extended Help

The functions mentioned in the implementation section were adjusted ad-hoc to deal with problems that emerged on the fly. 
Other problems may occur. In case you bump into one please open an issue on github.
"""
function build_nn_function(eq::EqT, sinput::Symbolics.Arr, nn::AbstractSymbolicNeuralNetwork)
    code = build_function(eq, sinput, nn.params...; expression = Val{true}) |> _reduce_code
    rewritten_code = fix_map_reduce(modify_input_arguments(rewrite_arguments(fix_create_array(code))))
    parallelized_code = make_kernel(rewritten_code)
    gen_fun = @RuntimeGeneratedFunction(parallelized_code)
    function (x::AbstractMatrix, ps) begin mapreduce(k -> gen_fun(x, ps, k), hcat, axes(x, 2)) end end
end

"""
    _reduce_code(code)

Reduce the code.

# Extended Help

For some reason `Symbolics.build_function` sometimes returns a tuple and sometimes it doesn't.
This function takes care of this.
"""
function _reduce_code(code::Expr)
    code
end

_reduce_code(code::Tuple) = code[1]

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

Change input arguments of type `(x, ps.L1, ps.L2)` etc to `(x, ps)`.
This should be used after [`rewrite_arguments`](@ref). Also see [`build_nn_function`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: modify_input_arguments

s = "(x, ps.L1, ps.L2, ps.L3)"
modify_input_arguments(s)

# output
"(x, ps)"
```
"""
function modify_input_arguments(s::AbstractString)
    @assert contains(s, "(x, ") "The first input argument must be x."
    regex = r"\(x, ps[a-zA-Z0-9., ]+\)"
    replace(s, regex => "(x, ps)")
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

s = "SymbolicUtils.Code.create_array(typeof(ˍ₋arg2)"
fix_create_array

# output
"SymbolicUtils.Code.create_array(Array
```
"""
function fix_create_array(s::AbstractString)
    @assert contains(s, "ˍ₋arg") "Doesn't contain ˍ₋arg!"
    # replace(s, r"\(SymbolicUtils\.Code\.create_array\)\(typeof\(..arg[0-9]+\), nothing, Val\{1\}\(\), Val\{\(2,\)\}\(\)," => "(")
    replace(s, r"\(SymbolicUtils\.Code\.create_array\)\(typeof\(..arg[0-9]+\)" => "SymbolicUtils.Code.create_array(Array")
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
"""
function fix_map_reduce(s::AbstractString)
    s1 = replace(s, "Symbolics._mapreduce" => "mapreduce")
    replace(s1, ", Colon(), (:init => false,)" => ", dims = Colon()")
end

function fix_map_reduce(expression::Expr)
    Meta.parse(fix_map_reduce(string(expression)))
end

"""
# Examples
```jldoctest
using SymbolicNeuralNetworks

s = "function (x, ps)\n begin\n getindex(x, 1) + getindex(x, 2) \n end\n end"
SymbolicNeuralNetworks.make_kernel(s)

# output

"function (x, ps, k)\n begin\n getindex(x, 1, k) + getindex(x, 2, k) \n end\n end"
```
"""
function make_kernel(s::AbstractString)
    # add k to function arguments
    s_added_k = replace(s, "function (x, ps)" => "function (x, ps, k)")
    # add k in body of function
    replace(s_added_k, r"getindex\(x, ([0-9]+)\)" => s"getindex(x, \1, k)")
end

function make_kernel(expression::Expr)
    Meta.parse(make_kernel(string(expression)))
end