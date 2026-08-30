# Generated functions whose parameter argument is one flat vector.
#
# A solver does not want a parameter set. A Newton iteration, a least-squares fit or a quasi-Newton
# method wants the unknowns as a vector and the derivative with respect to that vector as a matrix —
# which is what issue #21 asked for, from the nonlinear-integrator direction.
#
# The conversions themselves are not this package's to write: `NeuralNetworkParameters` has
# `flatten`/`unflatten` over a `ParameterLayout`, in both directions, allocation-free variants, and
# `ChainRulesCore` rules so that reverse mode goes through them. What is this package's is the
# symbolic half — generating a function that *takes* the flat form, and the derivative with respect to
# it as one flat object rather than as a nested set.

"""
    build_flat_function(eq, nn)
    build_flat_function(eq, nn, soutput)
    build_flat_function(eq, sparams, svariables...)

Turn a symbolic equation into an executable function whose parameter argument is a *flat vector*:

```julia
built_function(input, w)
built_function(input, output, w)
```

`w` may be a plain `AbstractVector` — in which case it is read through the layout of the parameters the
equation was built from — or a `NeuralNetworkParameters.FlatParameters`, which carries its own.

Takes the same keyword arguments as [`build_nn_function`](@ref), and accepts everything it accepts,
including a `NetworkParameters`.

The third form — the symbolic parameters and the symbolic data variables given directly rather than
taken from a network — is the one for degrees of freedom that are *not* a network's parameters:
nothing here reads a model. It is what [`flat_parameter_gradient`](@ref)'s second form pairs with.
Unlike that one it needs a `NetworkParameters` specifically, because
[`build_nn_function`](@ref) dispatches on one.

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: flatten
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
f = build_flat_function(c(snn.input, params(snn)), snn)

w, _ = flatten(params(nn))
f([1.0, 2.0], w) ≈ c([1.0, 2.0], params(nn))

# output

true
```

# Implementation

The flat vector is `unflatten`ed and the ordinary generated function called on the result, rather than
the code generation being changed to index into the vector. The conversion is a `copyto!` per leaf and
costs a fraction of a forward pass; generating a different kernel would save that and lose the ability
to hand the same equation a structured parameter set.

The conversion is `unflatten` and not `unflatten!` — it *allocates* the parameter set rather than
writing into the one the layout was built from, so `w` may have a different element type, which is
what lets `ForwardDiff` differentiate with respect to the flat form. This is unrelated to the
`inplace` keyword, which says how the generated kernel evaluates a batch and is passed through
untouched.
"""
function build_flat_function(eq, nn::AbstractSymbolicNeuralNetwork; kwargs...)
    build_flat_function(eq, params(nn), nn.input; kwargs...)
end

function build_flat_function(eq, nn::AbstractSymbolicNeuralNetwork, soutput::SymbolicVariables;
                             kwargs...)
    build_flat_function(eq, params(nn), nn.input, soutput; kwargs...)
end

function build_flat_function(eq, sparams::NetworkParameters, svariables::SymbolicVariables...;
                             kwargs...)
    f = build_nn_function(eq, sparams, svariables...; kwargs...)
    FlatParameterFunction{length(svariables)}(f, parameterlayout(sparams))
end

"""
    FlatParameterFunction{NDATA}(f, layout)

The function [`build_flat_function`](@ref) returns: it lays its flat parameter argument out into the
shape `layout` describes and calls `f` with it.
"""
struct FlatParameterFunction{NDATA, FT, LT} <: Function
    f::FT
    layout::LT
end

FlatParameterFunction{NDATA}(f::FT, layout::LT) where {NDATA, FT, LT} =
    FlatParameterFunction{NDATA, FT, LT}(f, layout)

(f::FlatParameterFunction{1})(input, w) = f.f(input, structured_parameters(f, w))
(f::FlatParameterFunction{2})(input, output, w) = f.f(input, output, structured_parameters(f, w))

"""
    structured_parameters(f, w)

The parameter set `w` stands for: `w` laid out according to `f`'s layout, or — when `w` carries a
layout of its own — according to that one.
"""
structured_parameters(f::FlatParameterFunction, w::AbstractVector) = unflatten(f.layout, w)
structured_parameters(::FlatParameterFunction, w::FlatParameters) = unflatten(w)

function Base.show(io::IO, f::FlatParameterFunction{NDATA}) where {NDATA}
    arguments = NDATA == 1 ? "(input, w)" : "(input, output, w)"
    print(io, "FlatParameterFunction ", arguments, " over ", flatlength(f.layout), " parameters")
end

# A method per shape rather than one signature over a union, as everywhere else in this release. The
# three answer different questions that share a body: a network carries its own parameters, a
# `NetworkParameters` is a set of degrees of freedom a network was evaluated at, and an
# [`EquationSet`](@ref) is one a caller wrote out. A union over `EquationSet` would additionally be a
# signature on `Base.NamedTuple`, which is permissible here only because the function is this
# package's — but saying which shape arrived costs one line each and is what the rest of the file does.
@doc raw"""
    flat_parameter_gradient(f, nn)
    flat_parameter_gradient(f, sparams)

Differentiate the symbolic expression `f` with respect to the parameters of `nn` — or with respect
to a set of symbolic parameters `sparams` given directly — as one flat object rather than as a set
nested like the parameters.

A scalar `f` gives a vector of length `flatlength(params(nn))`; an array-valued `f` gives the
``\mathrm{length}(f)\times\mathrm{flatlength}`` Jacobian

```math
J_{ij} = \frac{\partial{}f_i}{\partial{}w_j},
```

with the rows indexed by `vec(f)` — the convention [`Jacobian`](@ref) uses for the derivative with
respect to the input, and the matrix a Newton step is built from.

Pair it with [`build_flat_function`](@ref) for a function that is flat in both directions, and with
`NeuralNetworkParameters.unflatten` to read a column block of the result back as the entry of the
parameter set it belongs to.

The `sparams` form is the one for degrees of freedom that are not a network's parameters: nothing in
either function reads a model, so an expression over a set of symbolic leaves goes through both. As
with [`symbolic_parameter_gradient`](@ref), which this calls, `sparams` may be a `NetworkParameters`
or an [`EquationSet`](@ref) — the first is what a network's parameters are, the second what a caller
writes out. [`build_flat_function`](@ref), which this is paired with, is the narrower of the two and
wants a `NetworkParameters`, because [`build_nn_function`](@ref) dispatches on one for its
`sparams`.

# Examples

The gradient of a scalar expression, against the same derivative in its nested form:

```jldoctest
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_parameter_gradient
using AbstractNeuralNetworks: Chain, Dense, params
using NeuralNetworkParameters: flatten

c = Chain(Dense(2, 1, tanh))
snn = SymbolicNeuralNetwork(c)
scalar = sum(c(snn.input, params(snn)))

flat = flat_parameter_gradient(scalar, snn)
nested, _ = flatten(symbolic_parameter_gradient(scalar, snn))
(length(flat), all(isequal.(flat, nested)))

# output

(3, true)
```

The Jacobian of a vector-valued one:

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, params

c = Chain(Dense(2, 3, tanh))
snn = SymbolicNeuralNetwork(c)
size(flat_parameter_gradient(c(snn.input, params(snn)), snn))

# output

(3, 9)
```
"""
flat_parameter_gradient(f, dof::AbstractSymbolicNeuralNetwork) = _flat_parameter_gradient(f, dof)
flat_parameter_gradient(f, dof::NetworkParameters) = _flat_parameter_gradient(f, dof)
flat_parameter_gradient(f, dof::EquationSet) = _flat_parameter_gradient(f, dof)

_flat_parameter_gradient(f, dof) = flatten_gradient(symbolic_parameter_gradient(f, dof))

# Both shapes, for the same reason `flatten_equations` takes both: a gradient taken with respect to a
# network's parameters is a `NetworkParameters`, and one taken with respect to a caller's own set of
# symbolic variables has that set's shape.
"""
    flatten_gradient(gradient)

Lay a symbolic parameter derivative out flat: a single parameter-shaped set becomes a vector, and an
array of them — which is what differentiating an array-valued expression gives, one set per entry —
becomes a matrix with one row per entry. See [`flat_parameter_gradient`](@ref).
"""
flatten_gradient(gradient::NetworkParameters) = first(flatten_equations(gradient))
flatten_gradient(gradient::EquationSet) = first(flatten_equations(gradient))

function flatten_gradient(gradient::AbstractArray)
    isempty(gradient) && throw(ArgumentError(
        "cannot lay out the derivative of an empty expression."))
    stack(map(flatten_gradient, vec(gradient)); dims = 1)
end
