"""
    SymbolicExpression

Everything this package accepts as an *equation*: a single symbolic expression or an array of them.

Equations are built by applying a model to symbolic variables, e.g.

```julia
c(nn.input, params(nn))
```

Both representations `Symbolics` offers are covered. This package itself only ever produces arrays
of scalar expressions (see [`symbolic_variables`](@ref)), but a `Symbolics.Arr` handed in by a user
is accepted as well and normalised by [`scalar_expressions`](@ref).
"""
const SymbolicExpression = Union{Num, Symbolics.BasicSymbolic, AbstractArray{Num},
                                 AbstractArray{<:Symbolics.BasicSymbolic}, Symbolics.Arr{Num}}

"""
    EquationSet

An arbitrarily nested `NamedTuple` or `NetworkParameters`.

This is the shape of the parameters of a neural network, and therefore also the shape of a symbolic
derivative with respect to them. [`build_nn_function`](@ref) builds a whole set of equations of this
shape as one function; see [`flatten_equations`](@ref).
"""
const EquationSet = Union{NamedTuple, NetworkParameters}

"""
    scalar_expressions(eq)

Normalise an equation to scalar expressions, i.e. to a `Num` or an `Array{Num}`.

A `Symbolics.Arr` — the type `@variables x[1:n]` produces — is a symbolic object in its own right
whose entries are only materialised by `Symbolics.scalarize`. `Symbolics.build_function` cannot
generate code for one, so every equation passes through here on its way into
[`build_nn_function`](@ref). Arrays of `Num` (what this package builds, see
[`symbolic_variables`](@ref)) pass through unchanged.
"""
scalar_expressions(eq::SymbolicExpression) = _collect_if_array(Symbolics.scalarize(eq))

_collect_if_array(eq::Array) = eq
_collect_if_array(eq::AbstractArray) = collect(eq)
_collect_if_array(eq) = eq

# `Latexify` does not know how to print the activation function wrappers of `AbstractNeuralNetworks`,
# which show up in symbolic expressions that have not been expanded.
_latexraw(args::AbstractNeuralNetworks.GenericActivation; kwargs...) = _latexraw(args.σ; kwargs...)
_latexraw(args::AbstractNeuralNetworks.TanhActivation; kwargs...) = _latexraw(tanh; kwargs...)
