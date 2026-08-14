"""
    build_nn_function(eq, nn)

Build an executable function based on a symbolic equation, a symbolic input array and a [`SymbolicNeuralNetwork`](@ref).

This function can be called with:

```julia
built_function(input, ps)
```

# Keyword Arguments

- `cse`: perform *common subexpression elimination* when generating code (default `true`). See [`_build_nn_function`](@ref).
- `inplace`: evaluate a batch with an in-place kernel (default `true`). See below.

!!! warning "The default result cannot be differentiated with `Zygote`"
    With `inplace = true` the returned function allocates its result and lets the generated kernel
    *mutate* it, which `Zygote` does not support (`Mutating arrays is not supported`). Pass
    `inplace = false` to get the out-of-place version, which evaluates the kernel once per batch
    column and combines the results with `Base.reduce`; that one is differentiable, but allocates an
    array per column. Forward-mode AD (`ForwardDiff`) works with either, as the element type of the
    preallocated array is promoted over the inputs (see [`promoted_eltype`](@ref)).

# Implementation

Internally this is calling [`_build_nn_function_iip`](@ref) and then *parallelizing* the expression via the index `k`.
The kernel writes straight into a preallocated output array, so evaluating a batch costs a single allocation
instead of one array per column plus a `Base.reduce` fold.

For scalar-valued equations `Symbolics.build_function` does not emit an in-place form; those fall back to the
out-of-place [`_build_nn_function`](@ref), as does `inplace = false`.

# Extended Help

The functions mentioned in the implementation section were adjusted ad-hoc to deal with problems that emerged on the fly.
Other problems may occur. In case you bump into one please [open an issue on github](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues).
"""
function build_nn_function(eq::EqT, nn::AbstractSymbolicNeuralNetwork; cse::Bool = true, inplace::Bool = true)
    build_nn_function(eq, params(nn), nn.input; cse = cse, inplace = inplace)
end

function build_nn_function(
        eq::EqT, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr; reduce = hcat, cse::Bool = true, inplace::Bool = true)
    @assert ( (reduce == hcat) || (reduce == +) ) "Keyword reduce either has to be + or hcat!"
    sc_eq = Symbolics.scalarize(eq)
    # `Symbolics.build_function` emits no in-place form for a scalar equation, and generating the
    # code only to throw it away is not free, so do not even ask for it in that case.
    kernel! = (inplace && sc_eq isa AbstractArray) ?
              _build_nn_function_iip(sc_eq, sparams, sinput; reduce = reduce, cse = cse) : nothing
    isnothing(kernel!) && return _oop_batch_wrapper(_build_nn_function(sc_eq, sparams, sinput; cse = cse), reduce)
    _iip_batch_wrapper(kernel!, size(sc_eq), reduce)
end

"""
    _oop_batch_wrapper(gen_fun, reduce)

Evaluate the out-of-place kernel `gen_fun` once per batch column and combine the results.
Used for scalar-valued equations, for which there is no in-place kernel.
"""
function _oop_batch_wrapper(gen_fun, reduce)
    # Combine the per-column results in a single allocation. `mapreduce(…, hcat, …)` folds
    # `hcat` left-to-right, recopying the growing accumulator once per column (O(N²) in the
    # batch size); `Base.reduce(hcat, ::Vector)` sizes the result once instead (O(N)).
    gen_fun_returned(x, ps) = Base.reduce(reduce, [gen_fun(x, ps, k) for k in axes(x, 2)])
    function gen_fun_returned(x::Union{AbstractVector, Symbolics.Arr}, ps)
        output_not_reshaped = gen_fun(reshape(x, length(x), 1), ps, 1)
        # for vectors we do not reshape the output, as it may be a matrix
        output_not_reshaped
    end
    # check this! (definitely not correct in all cases!)
    function gen_fun_returned(x::AbstractArray{<:Number, 3}, ps)
        output_not_reshaped = gen_fun_returned(
            reshape(x, size(x, 1), size(x, 2) * size(x, 3)), ps)
        reshape(output_not_reshaped, size(output_not_reshaped, 1), size(x, 2), size(x, 3))
    end
    gen_fun_returned
end

"""
    _iip_batch_wrapper(kernel!, eq_size, reduce)

Allocate the output once and let the in-place `kernel!` write every batch column into it.
`eq_size` is the size of the (scalarized) equation, which fixes the shape of the result;
see [`allocate_batch_output`](@ref).
"""
function _iip_batch_wrapper(kernel!, eq_size::Tuple, reduce)
    function gen_fun_returned(x, ps)
        out = allocate_batch_output(promoted_eltype(x, ps), eq_size, size(x, 2), reduce)
        for k in axes(x, 2)
            kernel!(out, x, ps, k)
        end
        out
    end
    function gen_fun_returned(x::Union{AbstractVector, Symbolics.Arr}, ps)
        # for vectors we do not reshape the output, as it may be a matrix
        out = allocate_single_output(promoted_eltype(x, ps), eq_size, reduce)
        kernel!(out, reshape(x, length(x), 1), ps, 1)
        out
    end
    # check this! (definitely not correct in all cases!)
    function gen_fun_returned(x::AbstractArray{<:Number, 3}, ps)
        output_not_reshaped = gen_fun_returned(
            reshape(x, size(x, 1), size(x, 2) * size(x, 3)), ps)
        reshape(output_not_reshaped, size(output_not_reshaped, 1), size(x, 2), size(x, 3))
    end
    gen_fun_returned
end

"""
    promoted_eltype(args...)

The element type the generated code will produce, promoted over all inputs.

This is needed because the in-place kernels write into an array that we have to allocate
*before* calling them, so the element type cannot be inferred from a result. Promoting over
the inputs keeps `Float32` parameters, symbolic (`Num`) inputs and `ForwardDiff.Dual` numbers
working.

Note that this derives the element type from the *inputs*, not from the expression, which the
out-of-place path ([`_build_nn_function`](@ref)) does instead. The two can differ: an equation over
integer inputs and integer parameters evaluates to a `Float64`, which no `Array{Int}` can hold. The
allocators therefore widen an integer type with `float` (see [`allocate_batch_output`](@ref)). A
`Float32` network whose generated code contains a `Float64` constant is rounded to `Float32` rather
than widened, which is the behaviour one wants for the network but is worth being aware of.
"""
promoted_eltype(args...) = promote_type(map(_eltype, args)...)

_eltype(x::AbstractArray) = eltype(x)
_eltype(x::Number) = typeof(x)
_eltype(x::Tuple) = promote_type(map(_eltype, x)...)
_eltype(x::NamedTuple) = _eltype(values(x))
_eltype(x::NeuralNetworkParameters) = _eltype(values(x))

@doc raw"""
    allocate_batch_output(T, eq_size, batch_size, reduce)

Allocate the result of evaluating an equation of size `eq_size` over `batch_size` columns.

The shape matches what `Base.reduce(reduce, ::Vector)` over the per-column results used to
produce:
- `reduce = +`: the per-column results are summed, so the result has the size of the equation.
- `reduce = hcat`, vector-valued equation of length ``m``: an ``m\times{}N`` matrix.
- `reduce = hcat`, matrix-valued equation of size ``(m, n)``: the blocks are placed next to each
  other, giving an ``m\times(n\cdot{}N)`` matrix.

Equations of rank three or higher are *not* covered by that correspondence — `hcat` concatenates
those along their second dimension and keeps the third, whereas the linear indexing of the in-place
kernel flattens everything past the first dimension. No such equation arises in this package.

`T` is widened with `float` when it is an integer type: it comes from [`promoted_eltype`](@ref),
i.e. from the inputs, and an equation over integer inputs generally does not evaluate to an integer.
"""
allocate_batch_output(::Type{T}, eq_size::Tuple, ::Integer, ::typeof(+)) where {T} = zeros(_float_if_integer(T), eq_size...)
allocate_batch_output(::Type{T}, eq_size::Tuple{<:Integer}, batch_size::Integer, ::typeof(hcat)) where {T} = Array{_float_if_integer(T)}(undef, eq_size[1], batch_size)
allocate_batch_output(::Type{T}, eq_size::Tuple, batch_size::Integer, ::typeof(hcat)) where {T} = Array{_float_if_integer(T)}(undef, eq_size[1], prod(Base.tail(eq_size)) * batch_size)

"""
    allocate_single_output(T, eq_size, reduce)

Allocate the result of evaluating an equation of size `eq_size` for a single (vector) input.
Unlike [`allocate_batch_output`](@ref) this keeps the shape of the equation, as the output may
itself be a matrix.
"""
allocate_single_output(::Type{T}, eq_size::Tuple, ::typeof(+)) where {T} = zeros(_float_if_integer(T), eq_size...)
allocate_single_output(::Type{T}, eq_size::Tuple, ::typeof(hcat)) where {T} = Array{_float_if_integer(T)}(undef, eq_size...)

"""
    _float_if_integer(T)

`float(T)` for integer types, `T` for everything else. See [`allocate_batch_output`](@ref).
`Bool` is included, as `float(Bool)` is `Float64`.
"""
_float_if_integer(::Type{T}) where {T <: Integer} = float(T)
_float_if_integer(::Type{T}) where {T} = T

"""
    _build_nn_function(eq, params, sinput)

Build a function that can process a matrix. This is used as a starting point for [`build_nn_function`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: _build_nn_function, SymbolicNeuralNetwork
using AbstractNeuralNetworks: params, Chain, Dense, NeuralNetwork
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
eq = c(snn.input, params(snn))
built_function = _build_nn_function(eq, params(snn), snn.input)
built_function([1. 2.; 3. 4.], params(nn), 1)

# output

1-element Vector{Float64}:
 0.9912108161055604
```

Note that we have to supply an extra argument (index) to `_build_nn_function` that we do not have to supply to [`build_nn_function`](@ref).

# Keyword Arguments

- `cse`: perform *common subexpression elimination* (default `true`).

`Symbolics` stores an expression as a hash-consed *directed acyclic graph*, but
`Symbolics.build_function` prints it as a *tree*. Every time a subexpression is reused — the
output of layer ``n`` feeding each neuron of layer ``n+1``, or the forward pass shared by every
block of a symbolic pullback — the whole subtree is emitted again, so both the size of the
generated code and the amount of redundant arithmetic grow exponentially with the depth of the
network. With `cse = true` the graph is emitted as a `let` block of intermediate bindings
instead, which keeps code size proportional to the number of distinct nodes.

Pass `cse = false` to recover the old (fully inlined) output; this is mostly useful for
debugging, and for very small networks where the binding overhead is not amortized.

# Implementation

This first calls `Symbolics.build_function` with the keyword argument `expression = Val{true}` and then modifies the generated code by calling:
1. [`fix_create_array`](@ref),
2. [`rewrite_arguments`](@ref),
3. [`modify_input_arguments`](@ref),
4. [`fix_map_reduce`](@ref).

See the docstrings for those functions for details on how the code is modified.
"""
function _build_nn_function(
        eq::EqT, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr; cse::Bool = true)
    sc_eq = Symbolics.scalarize(eq)
    code = build_function_generated(_reduce, sc_eq, sinput, values(sparams)...; cse = cse)
    rewritten_code = fix_map_reduce(modify_input_arguments(rewrite_arguments(fix_create_array(code))))
    parallelized_code = make_kernel(rewritten_code)
    @RuntimeGeneratedFunction(parallelized_code)
end

"""
    _reduce(a)

Pick the *out-of-place* half of what `Symbolics.build_function` returns. It returns a
`(out_of_place, in_place)` tuple for array-valued equations and a single expression for
scalar-valued ones. See [`_reduce_iip`](@ref).
"""
_reduce(a) = a
_reduce(a::Tuple) = a[1]

"""
    _reduce_iip(a)

Pick the *in-place* half of what `Symbolics.build_function` returns, or `nothing` if there is
none (which is the case for scalar-valued equations). See [`_reduce`](@ref).
"""
_reduce_iip(::Any) = nothing
_reduce_iip(a::Tuple) = a[2]

"""
    build_function_generated(reducer, sc_eq, args...; cse)

Call `Symbolics.build_function` and pick the relevant half of its output with `reducer`
([`_reduce`](@ref) for the out-of-place form, [`_reduce_iip`](@ref) for the in-place one).

!!! note "Unsupported: reductions over un-scalarized symbolic arrays"
    Expressions containing an `arrayop` — a reduction over a `Symbolics.Arr` that has not been
    scalarized, e.g. `sum(c(input, ps))` — generate code that refers to variables nothing binds,
    so the resulting function throws an `UndefVarError` when it is called. This is independent
    of `cse` (the two modes just mangle different names). Reduce over `collect(c(input, ps))`
    instead.
"""
function build_function_generated(reducer, sc_eq, args...; cse::Bool)
    reducer(build_function(sc_eq, args...; expression = Val{true}, cse = cse))
end

@doc raw"""
    _build_nn_function_iip(eq, params, sinput; reduce, cse)

Build an *in-place* kernel that writes the result for batch column `k` into a preallocated array:

```julia
kernel!(out, input, ps, k)
```

Returns `nothing` for scalar-valued equations, for which `Symbolics.build_function` emits no
in-place form.

# Implementation

This works like [`_build_nn_function`](@ref), but keeps the second (in-place) half of what
`Symbolics.build_function` returns and post-processes it with:
1. [`fix_create_array`](@ref),
2. [`rewrite_arguments`](@ref) — unchanged, because the `ˍ₋out` argument is not counted in the
   `ˍ₋argN` numbering,
3. [`modify_input_arguments_iip`](@ref),
4. [`fix_map_reduce`](@ref),
5. [`make_kernel_iip`](@ref).

The in-place code addresses its output with a *linear* index (`ˍ₋out[i] = …`) whatever the shape
of the equation, which is what lets [`make_kernel_iip`](@ref) offset the writes by the batch
index instead of handing the kernel a view.
"""
function _build_nn_function_iip(
        eq::EqT, sparams::NeuralNetworkParameters, sinput::Symbolics.Arr; reduce = hcat, cse::Bool = true)
    sc_eq = Symbolics.scalarize(eq)
    code = build_function_generated(_reduce_iip, sc_eq, sinput, values(sparams)...; cse = cse)
    isnothing(code) && return nothing
    rewritten_code = fix_map_reduce(modify_input_arguments_iip(rewrite_arguments(fix_create_array(code))))
    parallelized_code = make_kernel_iip(rewritten_code, reduce, length(sc_eq))
    @RuntimeGeneratedFunction(parallelized_code)
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
    modify_input_arguments_iip(s)

Change input arguments of type `(ˍ₋out, sinput, ps.L1, ps.L2)` etc to `(ˍ₋out, sinput, ps)`.

This is the in-place counterpart of [`modify_input_arguments`](@ref) and should be used after
[`rewrite_arguments`](@ref). See [`_build_nn_function_iip`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: modify_input_arguments_iip

s = "(ˍ₋out, sinput, ps.L1, ps.L2, ps.L3)"
modify_input_arguments_iip(s)

# output
"(ˍ₋out, sinput, ps)"
```
"""
function modify_input_arguments_iip(s::AbstractString)
    @assert contains(s, "(ˍ₋out, sinput, ") "The first input arguments must be ˍ₋out and sinput."
    regex = r"\(ˍ₋out, sinput, ps[a-zA-Z0-9., ]+\)"
    replace(s, regex => "(ˍ₋out, sinput, ps)")
end

function modify_input_arguments_iip(expression::Expr)
    Meta.parse(modify_input_arguments_iip(string(expression)))
end

"""
   fix_create_array(s)

Fix a problem that occurs in connection with `create_array`.

The function `create_array` from `SymbolicUtils.Code` takes as first input the type of a symbolic array.
For reasons that are not entirely clear yet the first argument of `create_array` ends up being `ˍ₋arg2`, which is a `NamedTuple` of symoblic arrays.
We solve this problem by replacing `typeof(ˍ₋arg[0-9]+)` with `Array`, which is the most generic input to `create_array` and avoids `MethodError`s when `sinput` is a non-standard array type such as `ReshapedArray` or `SubArray`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: fix_create_array

s = "(SymbolicUtils.Code.create_array)(typeof(ˍ₋arg2)"
fix_create_array(s)

# output

"SymbolicUtils.Code.create_array(Array"
```

# Implementation

This is used for [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr)`](@ref) and [`_build_nn_function(::EqT, ::NeuralNetworkParameters, ::Symbolics.Arr, ::Symbolics.Arr)`](@ref).
"""
function fix_create_array(s::AbstractString)
    @assert contains(s, "ˍ₋arg") "Doesn't contain ˍ₋arg!"
    # replace(s, r"\(SymbolicUtils\.Code\.create_array\)\(typeof\(..arg[0-9]+\), nothing, Val\{1\}\(\), Val\{\(2,\)\}\(\)," => "(")
    replace(s,
        r"[\(]*SymbolicUtils\.Code\.create_array[\)]*\(typeof\(..arg[0-9]+\)" => "SymbolicUtils.Code.create_array(Array")
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

@doc raw"""
    make_kernel_iip(s, reduce, eq_length)

The in-place counterpart of [`make_kernel`](@ref): add the batch index `k` to the arguments,
index `sinput` with it, and redirect the writes to `ˍ₋out` so that a single preallocated array
can hold the result for the whole batch.

`eq_length` is the number of entries of the (scalarized) equation. The generated code assigns each
of them exactly once, which [`redirect_output_writes`](@ref) checks.

Which redirection is applied depends on how the per-column results are combined:
- `reduce = +`: the writes become `+=` and every column accumulates into the same buffer (which
  [`allocate_batch_output`](@ref) zeroes).
- `reduce = hcat`: the writes are shifted by ``(k - 1)\cdot\mathrm{eq\_length}``, which is exactly
  the offset of block `k` in the column-major layout of the concatenated result.

# Examples

```jldoctest
using SymbolicNeuralNetworks: make_kernel_iip

s = "function (ˍ₋out, sinput, ps)\n begin\n ˍ₋out[1] = getindex(sinput, 2)\n ˍ₋out[2] = getindex(sinput, 1) \n end\n end"
make_kernel_iip(s, hcat, 2)

# output

"function (ˍ₋out, sinput, ps, k)\n begin\n ˍ₋out[1 + (k - 1) * 2] = getindex(sinput, 2, k)\n ˍ₋out[2 + (k - 1) * 2] = getindex(sinput, 1, k) \n end\n end"
```

```jldoctest
using SymbolicNeuralNetworks: make_kernel_iip

s = "function (ˍ₋out, sinput, ps)\n begin\n ˍ₋out[1] = getindex(sinput, 2)\n ˍ₋out[2] = getindex(sinput, 1) \n end\n end"
make_kernel_iip(s, +, 2)

# output

"function (ˍ₋out, sinput, ps, k)\n begin\n ˍ₋out[1] += getindex(sinput, 2, k)\n ˍ₋out[2] += getindex(sinput, 1, k) \n end\n end"
```
"""
function make_kernel_iip(s::AbstractString, reduce, eq_length::Integer)
    # add k to function arguments
    s_added_k = replace(s, "function (ˍ₋out, sinput, ps)" => "function (ˍ₋out, sinput, ps, k)")
    # add k in body of function
    s_indexed = replace(s_added_k, r"getindex\(sinput, ([0-9]+)\)" => s"getindex(sinput, \1, k)")
    redirect_output_writes(s_indexed, reduce, eq_length)
end

function make_kernel_iip(expression::Expr, reduce, eq_length::Integer)
    Meta.parse(make_kernel_iip(string(expression), reduce, eq_length))
end

const OUTPUT_WRITE_REGEX = r"ˍ₋out\[([0-9]+)\] = "

"""
    redirect_output_writes(s, reduce, eq_length)

Rewrite the `ˍ₋out[i] = …` assignments of in-place generated code. See [`make_kernel_iip`](@ref).

# Implementation

Unlike the other rewrites in the pipeline this one cannot fail loudly on its own: a `replace` that
matches nothing returns the string unchanged, and the kernel then still compiles and runs. It would
just write every batch column into the same place (`reduce = hcat`) or overwrite instead of
accumulate (`reduce = +`), i.e. silently return wrong numbers. We therefore check that there is
exactly one assignment per entry of the equation, which is the property
`test/build_function/codegen_drift.jl` guards upstream.
"""
function redirect_output_writes(s::AbstractString, reduce, eq_length::Integer)
    n = length(collect(eachmatch(OUTPUT_WRITE_REGEX, s)))
    @assert n == eq_length "Expected $(eq_length) `ˍ₋out[i] = …` assignments in the generated code, found $(n). `Symbolics.build_function` has most likely changed how it emits in-place code; see test/build_function/codegen_drift.jl."
    _redirect_output_writes(s, reduce, eq_length)
end

_redirect_output_writes(s::AbstractString, ::typeof(+), ::Integer) = replace(s, OUTPUT_WRITE_REGEX => s"ˍ₋out[\1] += ")

function _redirect_output_writes(s::AbstractString, ::typeof(hcat), eq_length::Integer)
    replace(s, OUTPUT_WRITE_REGEX => SubstitutionString("ˍ₋out[\\1 + (k - 1) * $(eq_length)] = "))
end
