"""
    PARAMETER_NAME

The name the generated kernels give their parameter argument. The rewrite rules turn every
parameter argument of the code `Symbolics.build_function` emits into a `getproperty` on it, so that
the kernel can be called with a single `NeuralNetworkParameters`.
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
    DATA_NAMES

The names the generated kernels give their data arguments — the network input, and for the
two-argument form the target output as well.
"""
const DATA_NAMES = (:x1, :x2)

const RESERVED_NAMES = (PARAMETER_NAME, BATCH_INDEX, OUTPUT_NAME, DATA_NAMES...)

"""
    build_kernel(equation, sparams, svariables...; cse)

Build an *out-of-place* kernel that evaluates `equation` for batch column `k`:

```julia
kernel(x1, ps, k)          # one data argument
kernel(x1, x2, ps, k)      # two data arguments
```

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
function build_kernel(equation, sparams::NeuralNetworkParameters, svariables...; cse::Bool = true)
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
function build_kernel!(equation, sparams::NeuralNetworkParameters, svariables...;
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

function _collect_parameter_arguments!(paths, arrays, prefix::Tuple, sparams::EquationSet)
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
    body = substitute_symbols(body, argument_substitutions(generated_names, data_names, parameter_paths;
                                                           output_name = output_name))
    body = use_generic_array_constructor(body)
    body = use_base_mapreduce(body)
    index_by_batch(body, data_names)
end

function _data_names(svariables::Tuple)
    length(svariables) ≤ length(DATA_NAMES) || throw(ArgumentError(
        "at most $(length(DATA_NAMES)) data arguments are supported, got $(length(svariables))."))
    ntuple(i -> DATA_NAMES[i], length(svariables))
end

# The rewrite rules identify the arguments of the generated function by name, so a symbolic array
# that happens to carry one of the names the kernel uses itself would be rewritten twice.
function _assert_no_name_clash(generated_names)
    clashing = filter(in(RESERVED_NAMES), generated_names)
    isempty(clashing) || throw(ArgumentError(
        "the symbolic variables or parameters are named $(join(clashing, ", ")), which the " *
        "generated kernels use for their own arguments. Please rename them; " *
        "$(join(RESERVED_NAMES, ", ")) are reserved."))
    nothing
end
