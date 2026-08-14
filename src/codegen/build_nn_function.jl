"""
    SymbolicVariables

A symbolic array that a generated function takes as a data argument: the network input, or the
target output of a loss.

Both the `Vector{Num}` that [`SymbolicNeuralNetwork`](@ref) builds and the `Symbolics.Arr` that
`@variables x[1:n]` produces are accepted; the latter is scalarised by
[`scalar_expressions`](@ref).
"""
const SymbolicVariables = Union{AbstractVector{Num}, Symbolics.Arr}

"""
    build_nn_function(eq, nn)
    build_nn_function(eq, nn, soutput)

Turn a symbolic equation into an executable function.

The result is called with the network input and the network parameters, and with the target output
in between if the equation was built with one:

```julia
built_function(input, ps)
built_function(input, output, ps)
```

`input` may be a single sample (a vector), a batch (a matrix whose columns are the samples), or a
batch with two batch dimensions (a three-dimensional array). See [`AbstractBatchedFunction`](@ref)
for the shape of the result in each case.

    build_nn_function(eq, sparams, svariables...)

The same, but with the symbolic parameters and the symbolic data variables given explicitly rather
than taken from a [`SymbolicNeuralNetwork`](@ref).

# Keyword Arguments

- `cse`: perform *common subexpression elimination* when generating code (default `true`).
  See [`build_kernel`](@ref).
- `inplace`: evaluate a batch with an in-place kernel (default `true`). See below.
- `reduce`: how to combine the results of the individual samples of a batch, either `hcat`
  (default) or `+`.

!!! warning "The default result cannot be differentiated with `Zygote`"
    With `inplace = true` the returned function allocates its result and lets the generated kernel
    *mutate* it, which `Zygote` does not support (`Mutating arrays is not supported`). Pass
    `inplace = false` to get the out-of-place version, which is differentiable but allocates an
    array per sample. Forward-mode AD (`ForwardDiff`) works with either. See
    [`InPlaceBatchedFunction`](@ref) and [`OutOfPlaceBatchedFunction`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
built_function = build_nn_function(c(snn.input, params(snn)), snn)
built_function([1.0, 2.0], params(nn)) ≈ c([1.0, 2.0], params(nn))

# output

true
```

# Implementation

The equation is scalarised, a kernel that evaluates a single sample is generated from it
([`build_kernel`](@ref) or [`build_kernel!`](@ref)), and that kernel is wrapped in an
[`AbstractBatchedFunction`](@ref) which adds the batching. `Symbolics.build_function` emits no
in-place form for a scalar-valued equation, so those always take the out-of-place path.
"""
function build_nn_function(eq, nn::AbstractSymbolicNeuralNetwork; kwargs...)
    build_nn_function(eq, params(nn), nn.input; kwargs...)
end

function build_nn_function(eq, nn::AbstractSymbolicNeuralNetwork, soutput::SymbolicVariables; kwargs...)
    build_nn_function(eq, params(nn), nn.input, soutput; kwargs...)
end

function build_nn_function(eq::SymbolicExpression, sparams::NeuralNetworkParameters,
                           svariables::SymbolicVariables...;
                           reduce = hcat, cse::Bool = true, inplace::Bool = true)
    reduction = _check_reduction(reduce)
    equation = scalar_expressions(eq)
    variables = map(scalar_expressions, svariables)
    ndata = length(variables)

    kernel! = inplace ? build_kernel!(equation, sparams, variables...; reduction = reduction, cse = cse) : nothing
    isnothing(kernel!) || return InPlaceBatchedFunction{ndata}(kernel!, _equation_size(equation), reduction)

    kernel = build_kernel(equation, sparams, variables...; cse = cse)
    OutOfPlaceBatchedFunction{ndata}(kernel, _equation_size(equation), reduction)
end

function _check_reduction(reduce)
    (reduce === hcat || reduce === +) || throw(ArgumentError(
        "the keyword argument `reduce` has to be either `hcat` or `+`, got `$(reduce)`."))
    reduce
end

_equation_size(equation::AbstractArray) = size(equation)
_equation_size(::Any) = ()
