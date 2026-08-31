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

A keyed set of symbolic expressions: a `NamedTuple` whose values are expressions, arrays of them, or
further sets. It is what a caller writes by hand and hands to [`build_nn_function`](@ref) —

```julia
eqs = (a = c(nn.input, params(nn)), b = c(nn.input, params(nn)) .^ 2)
```

— and it is **not** a set of parameters, which is a
[`NeuralNetworkParameters.NetworkParameters`](https://juliagni.github.io/NeuralNetworkParameters.jl/stable/representations/).
A plain link and not an `@extref`: this package configures no `DocumenterInterLinks` inventories, so
an `@extref` here has nothing to resolve against and fails the build at `CrossReferences`. The two
share a shape and nothing else: a
parameter set is the thing a network is evaluated *at*, and an equation set is a bundle of expressions
that happen to be keyed. Naming them apart is what keeps a signature honest about which it wants; the
one place both arrive is a symbolic *gradient*, which is parameter-shaped and holds expressions, and
[`flatten_equations`](@ref) has a method for each.

An alias for `NamedTuple` rather than a type of its own, so a method taking one is a method on
`Base.NamedTuple` — permissible only because the functions here are this package's. Do not extend a
foreign generic on it.
"""
const EquationSet = NamedTuple

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
function _latexraw(args::AbstractNeuralNetworks.GenericActivation; kwargs...)
    _latexraw(args.σ; kwargs...)
end
function _latexraw(args::AbstractNeuralNetworks.TanhActivation; kwargs...)
    _latexraw(tanh; kwargs...)
end
