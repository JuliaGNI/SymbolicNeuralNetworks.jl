@doc raw"""
    Gradient <: Derivative

Computes and stores the derivative of a symbolic expression with respect to the *parameters* of a
[`SymbolicNeuralNetwork`](@ref).

# Constructors

    Gradient(f, nn)

Differentiate the symbolic `f` with respect to the parameters of `nn`.

    Gradient(nn)

Differentiate the symbolic output of `nn`, i.e. `nn.model(nn.input, params(nn))`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
(Gradient(nn) |> derivative)[1].L1.b

# output

1-element Vector{Symbolics.Num}:
 1 - (tanh(W_2₁ + W_1₁ˏ₁*x₁ + W_1₁ˏ₂*x₂)^2)
```

# Implementation

Internally this uses [`symbolic_parameter_gradient`](@ref). For an array-valued `f` the result is an
array of the same shape whose entries are the parameter-shaped gradients of the corresponding entry
of `f` — so the gradient of a matrix is a matrix of `NetworkParameters`, each of which is the
ordinary gradient of one matrix element.
"""
struct Gradient{OT, SDT, ST} <: Derivative{OT, SDT, ST}
    f::OT
    ∇::SDT
    nn::ST
end

"""
    derivative(g)

The symbolic gradient stored in `g`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative, symbolic_parameter_gradient
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
g = Gradient(nn)

isequal(derivative(g), symbolic_parameter_gradient(g.f, nn))

# output

true
```
"""
derivative(g::Gradient) = g.∇

function Gradient(f, nn::SymbolicNeuralNetwork)
    f isa AbstractArray || @warn "You should only use `Gradient` together with array expressions! Maybe you wanted to use `SymbolicPullback`."
    Gradient(f, symbolic_parameter_gradient(f, nn), nn)
end

Gradient(nn::SymbolicNeuralNetwork) = Gradient(nn.model(nn.input, params(nn)), nn)

@doc raw"""
    symbolic_parameter_gradient(f, nn)
    symbolic_parameter_gradient(f, sparams)

Differentiate the symbolic expression `f` with respect to the parameters of `nn`, or with respect to
a set of symbolic parameters `sparams` given directly.

The result has the same nesting as the parameters of `nn`. For an array-valued `f` it is an array of
such parameter sets, one per entry of `f`.

The second form is what the degrees of freedom of an expression that is not a network's forward pass
go through: nothing here reads the model, only the parameters. `sparams` is anything
[`symbolic_differentials`](@ref) can walk, i.e. a `NetworkParameters` or a nested `NamedTuple` of
symbolic leaves — the same `ParameterSet` the parameters of a
[`SymbolicNeuralNetwork`](@ref) are allowed to be.

This is used by [`Gradient`](@ref) and by [`SymbolicPullback`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, symbolic_parameter_gradient
using AbstractNeuralNetworks
using AbstractNeuralNetworks: params

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
symbolic_parameter_gradient(c(nn.input, params(nn)), nn)[1].L1.b

# output

1-element Vector{Symbolics.Num}:
 1 - (tanh(W_2₁ + W_1₁ˏ₁*x₁ + W_1₁ˏ₂*x₂)^2)
```
"""
function symbolic_parameter_gradient(f, nn::AbstractSymbolicNeuralNetwork)
    symbolic_parameter_gradient(f, params(nn))
end

function symbolic_parameter_gradient(f, sparams::ParameterSet)
    differentials = symbolic_differentials(sparams)
    _parameter_gradient(scalar_expressions(f), differentials)
end

_parameter_gradient(f::AbstractArray, differentials) = [symbolic_derivative(entry, differentials) for entry in f]
_parameter_gradient(f, differentials) = symbolic_derivative(f, differentials)
