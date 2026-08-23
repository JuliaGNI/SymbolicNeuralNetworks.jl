# Rewrite rules for the code that `Symbolics.build_function` emits.
#
# `Symbolics.build_function(…; expression = Val{true})` returns an `Expr` for a function that takes
# one argument per symbolic array it was given and evaluates the equation for a *single* sample.
# What this package needs instead is a function that takes the parameters as one nested object and
# evaluates one column of a batch. The rules below bridge that gap; `src/codegen/kernels.jl` applies
# them in order.
#
# All of them work on the syntax tree, never on its printed form. Two properties of the emitted
# code make that necessary rather than merely nicer:
#
#   * `SymbolicUtils` embeds function *objects* in the tree, so the callee of `(getindex)(x, 1)` is
#     `Base.getindex` itself and not the symbol `:getindex`. A rule that matches only the symbol
#     silently does nothing, and — because the batch index it fails to add defaults to the first
#     column — the result is still *correct for the first sample*. See [`callee_name`](@ref).
#   * Argument names such as `ˍ₋arg2` are an implementation detail of Symbolics, and the name of the
#     first argument is whatever the user happened to call their symbolic array. Matching arguments
#     by *position* (see [`argument_substitutions`](@ref)) depends on neither.

"""
    postwalk(f, x)

Apply `f` to every node of the expression tree `x`, children first. Nodes that `f` returns are not
visited again, so a rule may insert new syntax without it being rewritten in turn.
"""
postwalk(f, x) = f(x)
postwalk(f, e::Expr) = f(Expr(e.head, map(arg -> postwalk(f, arg), e.args)...))

"""
    callee_name(e)

The name of the function a call expression calls, or `nothing` if `e` is not a call.

The emitted code refers to a function either by symbol, by function object, or by a qualified path
(`SymbolicUtils.Code.create_array`), depending on how it was constructed. All three forms are
reduced to a plain `Symbol` here so that the rules only have to deal with one of them.
"""
callee_name(::Any) = nothing
function callee_name(e::Expr)
    e.head === :call || return nothing
    _name_of(first(e.args))
end

_name_of(callee::Symbol) = callee
_name_of(callee::Function) = nameof(callee)
_name_of(callee::GlobalRef) = callee.name
_name_of(callee::Expr) = callee.head === :. && callee.args[2] isa QuoteNode ? callee.args[2].value : nothing
_name_of(::Any) = nothing

"""
    function_arguments_and_body(expr)

Split a generated function expression into its vector of argument names and its body.

Throws an `ArgumentError` if `expr` is not of the shape `Symbolics.build_function` is documented to
return, which turns an upstream change into an error at code-generation time rather than into
subtly wrong code.
"""
function function_arguments_and_body(expr::Expr)
    expr.head === :function || throw(ArgumentError(
        "expected `Symbolics.build_function` to return a function definition, got an expression with head `$(expr.head)`."))
    signature = expr.args[begin]
    (signature isa Expr && signature.head === :tuple) || throw(ArgumentError(
        "expected the generated function to have an anonymous argument tuple, got `$(signature)`."))
    signature.args, expr.args[end]
end

"""
    function_expression(argument_names, body)

Assemble an anonymous function definition; the inverse of [`function_arguments_and_body`](@ref).
"""
function_expression(argument_names, body) = Expr(:function, Expr(:tuple, argument_names...), body)

@doc raw"""
    argument_substitutions(generated_names, data_names, parameter_paths; output_name)

Map the argument names of the generated function onto the names the kernel uses, *by position*.

`Symbolics.build_function` was handed the data variables first and then one argument per parameter
array (see [`parameter_arguments`](@ref)), so with `output_name` set — the in-place form prepends an
output argument — the correspondence is

```
(ˍ₋out, ˍ₋arg1, …, ˍ₋arg_ndata, ˍ₋arg_{ndata+1}, …)  ↦  (out, x1, …, x_ndata, ps.L1.W, …)
```

Every parameter argument becomes a chain of `getproperty` calls on the single parameter argument
`ps`, which is what lets the kernel be called with `NetworkParameters` instead of a flat
argument list.

# Examples

```jldoctest
using SymbolicNeuralNetworks: argument_substitutions

substitutions = argument_substitutions([:ˍ₋arg1, :ˍ₋arg2], (:x1,), (((:L1, :W)),); output_name = nothing)
(substitutions[:ˍ₋arg1], substitutions[:ˍ₋arg2])

# output

(:x1, :(ps.L1.W))
```
"""
function argument_substitutions(generated_names::AbstractVector, data_names::Tuple,
                                parameter_paths::Tuple; output_name::Union{Symbol, Nothing})
    expected = length(data_names) + length(parameter_paths) + (isnothing(output_name) ? 0 : 1)
    length(generated_names) == expected || throw(ArgumentError(
        "the generated function takes $(length(generated_names)) arguments, expected $(expected)."))

    substitutions = Dict{Symbol, Any}()
    offset = 0
    if !isnothing(output_name)
        substitutions[generated_names[1]] = output_name
        offset = 1
    end
    for (i, name) in enumerate(data_names)
        substitutions[generated_names[offset + i]] = name
    end
    offset += length(data_names)
    for (i, path) in enumerate(parameter_paths)
        substitutions[generated_names[offset + i]] = access_expression(PARAMETER_NAME, path)
    end
    substitutions
end

"""
    access_expression(name, path)

The expression that reads `path` out of the object called `name`, e.g. `ps.L1.W` for `(:L1, :W)`.
"""
access_expression(name::Symbol, path::Tuple) = foldl((e, key) -> Expr(:., e, QuoteNode(key)), path; init = name)

"""
    substitute_symbols(expr, substitutions)

Replace every symbol of `expr` that is a key of `substitutions` by the corresponding value.
"""
function substitute_symbols(expr, substitutions::Dict{Symbol, Any})
    postwalk(expr) do node
        node isa Symbol ? get(substitutions, node, node) : node
    end
end

"""
    use_generic_array_constructor(expr)

Replace `SymbolicUtils.Code.create_array(typeof(…), …)` with `create_array(Array, …)`.

`create_array` takes the array type to construct as its first argument, and `Symbolics` fills that
in with the type of one of the *arguments* of the generated function. For us that argument is the
parameter set — a `NamedTuple`, from which no array can be constructed — and even where it is an
array it may be a `SubArray` or a `ReshapedArray` that `create_array` has no method for. `Array` is
the generic choice that works in every case.

# Examples

```jldoctest
using SymbolicNeuralNetworks: use_generic_array_constructor

use_generic_array_constructor(:((SymbolicUtils.Code.create_array)(typeof(ps), nothing, Val{1}(), a)))

# output

:(SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), a))
```
"""
function use_generic_array_constructor(expr)
    postwalk(expr) do node
        if callee_name(node) === :create_array && length(node.args) > 1 && callee_name(node.args[2]) === :typeof
            Expr(:call, node.args[1], :Array, node.args[3:end]...)
        else
            node
        end
    end
end

"""
    use_base_mapreduce(expr)

Replace `Symbolics._mapreduce` with `Base.mapreduce`.

`Symbolics._mapreduce` cannot be differentiated by `Zygote`, whereas `Base.mapreduce` can. Its
trailing `Colon(), (:init => false,)` arguments are the positional form of `dims = Colon()`.

Nothing this package generates contains a `Symbolics._mapreduce` any more — that used to come from
reductions over un-scalarised `Symbolics.Arr`s, which [`scalar_expressions`](@ref) now rules out.
The rule is kept because the equations a *user* passes in are not under our control.

# Examples

```jldoctest
using SymbolicNeuralNetworks: use_base_mapreduce

use_base_mapreduce(:(Symbolics._mapreduce(identity, +, x, Colon(), (:init => false,))))

# output

:(mapreduce(identity, +, x; dims = Colon()))
```
"""
function use_base_mapreduce(expr)
    postwalk(expr) do node
        if callee_name(node) === :_mapreduce && length(node.args) ≥ 4
            Expr(:call, :mapreduce, Expr(:parameters, Expr(:kw, :dims, :(Colon()))),
                 node.args[2:(end - 2)]...)
        else
            node
        end
    end
end

"""
    index_by_batch(expr, data_names)

Turn `x[i]` into `x[i, k]` for every data argument `x`.

The generated code reads a data argument as if it were a single sample. Adding the batch index `k`
as a second index is what makes the same code read column `k` of a matrix instead, and is the reason
the kernels take a batch index at all.

Both forms `Symbolics` emits for reading an entry are handled: `x[i]` (`Expr(:ref, …)`) and
`getindex(x, i)`, which it uses for arguments that were `Symbolics.Arr`s.

Throws an `ArgumentError` if a data argument is used other than by indexing it with a single index,
because then the batch dimension would silently be ignored for that use.

# Examples

```jldoctest
using SymbolicNeuralNetworks: index_by_batch

index_by_batch(:(x1[1] + getindex(x1, 2)), (:x1,))

# output

:(x1[1, k] + getindex(x1, 2, k))
```
"""
function index_by_batch(expr, data_names::Tuple)
    postwalk(expr) do node
        if node isa Expr && node.head === :ref && length(node.args) == 2 && node.args[1] ∈ data_names
            Expr(:ref, node.args[1], node.args[2], BATCH_INDEX)
        elseif callee_name(node) === :getindex && length(node.args) == 3 && node.args[2] ∈ data_names
            Expr(:call, node.args[1], node.args[2], node.args[3], BATCH_INDEX)
        else
            _assert_indexed_only(node, data_names)
            node
        end
    end
end

# A data argument that survives anywhere but as the object of a single-index read would be evaluated
# for the whole batch at once instead of for one sample; `:ref` and `:.` nodes are where the
# rewrites above and the parameter access legitimately leave a bare symbol behind.
function _assert_indexed_only(node, data_names::Tuple)
    (node isa Expr && node.head ∈ (:call, :ref, :.)) || return
    arguments = node.head === :call ? node.args[2:end] : node.args
    for argument in arguments
        argument isa Symbol && argument ∈ data_names && throw(ArgumentError(
            "the generated code uses the data argument `$(argument)` other than by indexing a " *
            "single sample from it (in `$(node)`), so it cannot be evaluated over a batch. This " *
            "usually means the equation contains an operation on an un-scalarised symbolic array."))
    end
end

@doc raw"""
    accumulate_into_output(expr, output_name, reduction, equation_length)

Rewrite the `out[i] = …` assignments of in-place generated code so that a single preallocated array
can hold the result of a whole batch.

Which rewrite applies depends on how the per-sample results are combined:
- `reduction = +`: the writes become `+=`, so every sample accumulates into the same buffer (which
  [`allocate_batch_output`](@ref) zeroes).
- `reduction = hcat`: the writes are shifted by ``(k - 1)\cdot\mathrm{equation\_length}``, which is
  the offset of block `k` in the column-major layout of the concatenated result.

The generated code addresses its output with a single *linear* index whatever the shape of the
equation, which is what makes that offset arithmetic correct. `equation_length` assignments are
expected, one per entry of the equation; anything else throws, because a `replace` that matches
nothing would leave a kernel that still compiles and runs but writes every sample to the same place.

# Examples

```jldoctest
using SymbolicNeuralNetworks: accumulate_into_output

accumulate_into_output(:(out[1] = a), :out, hcat, 1)

# output

:(out[1 + (k - 1) * 1] = a)
```

```jldoctest
using SymbolicNeuralNetworks: accumulate_into_output

accumulate_into_output(:(out[1] = a), :out, +, 1)

# output

:(out[1] += a)
```
"""
function accumulate_into_output(expr, output_name::Symbol, reduction, equation_length::Integer)
    writes = Ref(0)
    rewritten = postwalk(expr) do node
        if _is_output_write(node, output_name)
            writes[] += 1
            _redirect_write(node, output_name, reduction, equation_length)
        else
            node
        end
    end
    writes[] == equation_length || throw(ArgumentError(
        "expected $(equation_length) `$(output_name)[i] = …` assignments in the generated code, " *
        "found $(writes[]). `Symbolics.build_function` has most likely changed how it emits " *
        "in-place code; see `test/codegen/codegen_drift.jl`."))
    rewritten
end

function _is_output_write(node, output_name::Symbol)
    node isa Expr && node.head === :(=) && node.args[1] isa Expr &&
        node.args[1].head === :ref && node.args[1].args[1] === output_name
end

_redirect_write(node::Expr, ::Symbol, ::typeof(+), ::Integer) = Expr(:+=, node.args[1], node.args[2])

function _redirect_write(node::Expr, output_name::Symbol, ::typeof(hcat), equation_length::Integer)
    index = node.args[1].args[2]
    Expr(:(=), Expr(:ref, output_name, :($index + ($BATCH_INDEX - 1) * $equation_length)), node.args[2])
end
