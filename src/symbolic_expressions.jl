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

# `ParameterSet` used to be defined here, as `Union{NamedTuple, NetworkParameters}` -- the shape of the
# parameters of a neural network, and therefore also the shape of a symbolic derivative with respect to
# them. It is `NeuralNetworkParameters.ParameterSet` now, which is the same union in the package that
# owns the type: this one had it under a name of its own, `AbstractNeuralNetworks` and
# `GeometricMachineLearning` were spelling it out inline, and `GeometricOptimizers` had a third name for
# a narrower version. What this package called an equation set is a parameter set that happens to hold
# equations, so nothing is lost by naming it after the shape.

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
