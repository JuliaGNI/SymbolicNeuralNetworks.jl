"""
    PARAMETER_NAME

The name the generated kernels give their parameter argument. The rewrite rules turn every
parameter argument of the code `Symbolics.build_function` emits into a `getproperty` on it, so that
the kernel can be called with a single `NetworkParameters`.
"""
const PARAMETER_NAME = :ps

"""
    BATCH_INDEX

The name the generated kernels give the index of the batch column they evaluate.
See [`index_by_batch`](@ref).
"""
const BATCH_INDEX = :k

"""
    OUTPUT_NAME

The name the generated in-place kernels give the array they write into.
See [`accumulate_into_output`](@ref).
"""
const OUTPUT_NAME = :out

"""
    data_name(i)

The name the generated kernels give their `i`-th data argument: `x1`, `x2`, … There is no bound on
how many there may be. One is the network input; a second is typically the target output of a loss;
the layerwise pullback uses one per entry of a layer's seam plus one for the output sensitivities
(see [`seam_interface`](@ref)).
"""
data_name(i::Integer) = Symbol(:x, i)

"""
    FIXED_NAMES

The three names a generated kernel always gives its own arguments, whatever its arity.
See [`is_reserved_name`](@ref) for the rest.
"""
const FIXED_NAMES = (PARAMETER_NAME, BATCH_INDEX, OUTPUT_NAME)

"""
    is_reserved_name(name)

Whether `name` is one a generated kernel gives an argument of its own, and therefore one a symbolic
variable left *free* in an equation may not carry.

That is [`FIXED_NAMES`](@ref) together with the whole `x1`, `x2`, … family of [`data_name`](@ref)s —
the family and not just the arities in use, because whether a given name is generated would otherwise
depend on how many data arguments the equation happens to have, and a free variable named `x3` would
pass the check today and break the day a third data argument arrived.
"""
is_reserved_name(name::Symbol) = name ∈ FIXED_NAMES || _is_data_name(String(name))

function _is_data_name(name::AbstractString)
    length(name) > 1 && name[1] == 'x' && name[2] != '0' && all(isdigit, @view name[2:end])
end

const RESERVED_NAMES_MESSAGE = "$(join(FIXED_NAMES, ", ")) and x1, x2, … are reserved"

"""
    build_kernel(equation, sparams, svariables...; cse)

Build an *out-of-place* kernel that evaluates `equation` for batch column `k`:

```julia
kernel(x1, ps, k)          # one data argument
kernel(x1, x2, ps, k)      # two data arguments
```

There is no bound on the number of data arguments; see [`data_name`](@ref).

`sparams` are the symbolic parameters and `svariables` the symbolic data variables the equation was
built from. See [`build_kernel!`](@ref) for the in-place counterpart and
[`build_nn_function`](@ref) for the function that batching, allocation and reshaping are added to.

# Examples

```jldoctest
using SymbolicNeuralNetworks: build_kernel, SymbolicNeuralNetwork
using AbstractNeuralNetworks: params, Chain, Dense, NeuralNetwork
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
kernel = build_kernel(c(snn.input, params(snn)), params(snn), snn.input)
kernel([1.0 2.0; 3.0 4.0], params(nn), 1)

# output

1-element Vector{Float64}:
 0.9912108161055604
```

# Keyword Arguments

- `cse`: perform *common subexpression elimination* when generating code (default `true`).

`Symbolics` stores an expression as a hash-consed *directed acyclic graph* but
`Symbolics.build_function` prints it as a *tree*. Every time a subexpression is reused — the output
of layer ``n`` feeding each neuron of layer ``n+1``, or the forward pass shared by every block of a
symbolic gradient — the whole subtree is emitted again, so both the size of the generated code and
the amount of redundant arithmetic grow exponentially with the depth of the network. With
`cse = true` the graph is emitted as a `let` block of intermediate bindings instead, which keeps the
code size proportional to the number of distinct nodes.

Pass `cse = false` to recover the fully inlined output; that is mostly useful for debugging, and for
very small networks where the binding overhead is not amortised.
"""
function build_kernel(equation, sparams::NetworkParameters, svariables...; cse::Bool = true)
    paths, arrays = parameter_arguments(sparams)
    expression = generated_expression(equation, svariables, arrays; inplace = false, cse = cse)
    data_names = _data_names(svariables)
    body = _rewrite_body(expression, data_names, paths, nothing)
    @RuntimeGeneratedFunction(function_expression((data_names..., PARAMETER_NAME, BATCH_INDEX), body))
end

@doc raw"""
    build_kernel!(equation, sparams, svariables...; reduction, cse)

Build an *in-place* kernel that writes the result for batch column `k` into a preallocated array:

```julia
kernel!(out, x1, ps, k)          # one data argument
kernel!(out, x1, x2, ps, k)      # two data arguments
```

Where in `out` the result goes depends on `reduction`; see [`accumulate_into_output`](@ref).

Returns `nothing` for a scalar-valued equation, for which `Symbolics.build_function` emits no
in-place form.

Evaluating a batch with such a kernel costs a single allocation instead of one array per column
plus a `Base.reduce` fold, but the result is produced by mutation and can therefore not be
differentiated by `Zygote`. See [`build_kernel`](@ref) for the keyword arguments.
"""
function build_kernel!(equation, sparams::NetworkParameters, svariables...;
                       reduction, cse::Bool = true)
    paths, arrays = parameter_arguments(sparams)
    expression = generated_expression(equation, svariables, arrays; inplace = true, cse = cse)
    isnothing(expression) && return nothing
    data_names = _data_names(svariables)
    body = _rewrite_body(expression, data_names, paths, OUTPUT_NAME)
    body = accumulate_into_output(body, OUTPUT_NAME, reduction, length(equation))
    @RuntimeGeneratedFunction(function_expression((OUTPUT_NAME, data_names..., PARAMETER_NAME, BATCH_INDEX), body))
end

"""
    generated_expression(equation, svariables, sarrays; inplace, cse)

Call `Symbolics.build_function` and pick the half of its output that is asked for.

It returns an `(out_of_place, in_place)` pair for an array-valued equation and a single expression
for a scalar-valued one; `nothing` is returned when the in-place half is asked for and there is
none.
"""
function generated_expression(equation, svariables::Tuple, sarrays::Tuple; inplace::Bool, cse::Bool)
    code = Symbolics.build_function(equation, svariables..., sarrays...; expression = Val{true}, cse = cse)
    inplace ? _in_place_half(code) : _out_of_place_half(code)
end

"""
    parameter_arguments(sparams)

Flatten a nested parameter set into the flat list of symbolic arrays that
`Symbolics.build_function` is handed, together with the access path of each within the parameter
object.

`Symbolics.build_function` only recognises a symbolic array that is passed to it *as an argument*,
so passing the nested parameter object as a whole would leave its entries as free variables in the
generated code. Flattening here and rebuilding the access paths in
[`argument_substitutions`](@ref) keeps the kernel's own interface nested regardless.

# Examples

```jldoctest
using SymbolicNeuralNetworks: parameter_arguments, SymbolicNeuralNetwork
using AbstractNeuralNetworks: Chain, Dense, params

snn = SymbolicNeuralNetwork(Chain(Dense(2, 1, tanh)))
first(parameter_arguments(params(snn)))

# output

((:L1, :W), (:L1, :b))
```
"""
function parameter_arguments(sparams)
    paths = Tuple[]
    arrays = Any[]
    _collect_parameter_arguments!(paths, arrays, (), sparams)
    Tuple(paths), Tuple(arrays)
end

# This one recurses through `keys`/`getindex` itself rather than through `mapparameters`, so it meets a
# container's *layers* — plain `NamedTuple`s — on the way down and needs a method for each shape.
function _collect_parameter_arguments!(paths, arrays, prefix::Tuple, sparams::NetworkParameters)
    for key in keys(sparams)
        _collect_parameter_arguments!(paths, arrays, (prefix..., key), sparams[key])
    end
end

function _collect_parameter_arguments!(paths, arrays, prefix::Tuple, sparams::NamedTuple)
    for key in keys(sparams)
        _collect_parameter_arguments!(paths, arrays, (prefix..., key), sparams[key])
    end
end

function _collect_parameter_arguments!(paths, arrays, prefix::Tuple, sarray)
    push!(paths, prefix)
    push!(arrays, sarray)
end

_out_of_place_half(code) = code
_out_of_place_half(code::Tuple) = code[begin]
_in_place_half(::Any) = nothing
_in_place_half(code::Tuple) = code[end]

"""
    _rewrite_body(expression, data_names, parameter_paths, output_name)

Apply the rewrite rules of `src/codegen/expression_rewriting.jl` to the body of a generated
function, in the order they depend on each other: the arguments are renamed first so that the later
rules can recognise the data arguments, the array constructor is fixed before the batch index is
added so that the `typeof(…)` it contains is not mistaken for a use of a data argument.
"""
function _rewrite_body(expression::Expr, data_names::Tuple, parameter_paths::Tuple,
                       output_name::Union{Symbol, Nothing})
    generated_names, body = function_arguments_and_body(expression)
    _assert_no_name_clash(generated_names)
    _assert_no_reserved_names_in_body(body)
    body = substitute_symbols(body, argument_substitutions(generated_names, data_names, parameter_paths;
                                                           output_name = output_name))
    body = use_generic_array_constructor(body)
    body = use_base_mapreduce(body)
    index_by_batch(body, data_names)
end

_data_names(svariables::Tuple) = ntuple(data_name, length(svariables))

# The rewrite rules identify the arguments of the generated function by name, so a symbolic array
# that happens to carry one of the names the kernel uses itself would be rewritten twice.
function _assert_no_name_clash(generated_names)
    clashing = filter(is_reserved_name, generated_names)
    isempty(clashing) || throw(ArgumentError(
        "the symbolic variables or parameters are named $(join(clashing, ", ")), which the " *
        "generated kernels use for their own arguments. Please rename them; " *
        RESERVED_NAMES_MESSAGE * "."))
    nothing
end

"""
    _assert_no_reserved_names_in_body(body)

Reject a generated body that already contains one of the names the kernels give their own arguments.

A symbolic variable that is passed to `Symbolics.build_function` becomes an argument and is renamed
to `ˍ₋argN`, but one that is *not* — a variable left free in the equation, i.e. neither a data
variable nor a parameter — survives into the body under its own name. If that name happens to be
`k`, the kernel's batch index binds it and the equation silently evaluates with the column number in
place of the variable; if it is `ps`, it binds the parameter set. Neither is caught by
`_assert_no_name_clash`, which only sees the argument names.

The check has to run *before* the arguments are substituted, since afterwards the reserved names are
all over the body legitimately. At that point the only symbols in the tree are `ˍ₋argN`,
`var"##cse#N"` and literals — functions are embedded as objects — so a reserved name can only have
come from a free variable.
"""
function _assert_no_reserved_names_in_body(body)
    found = Symbol[]
    postwalk(body) do node
        node isa Symbol && is_reserved_name(node) && node ∉ found && push!(found, node)
        node
    end
    isempty(found) || throw(ArgumentError(
        "the generated code refers to $(join(found, ", ")), which the generated kernels use for " *
        "their own arguments. This means the equation contains a symbolic variable of that name " *
        "that was passed neither as a data variable nor as a parameter. Please rename it; " *
        RESERVED_NAMES_MESSAGE * "."))
    nothing
end
