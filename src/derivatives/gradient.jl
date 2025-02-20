@doc raw"""
    Gradient <: Derivative

Computes and stores the gradient of a symbolic function with respect to the parameters of a [`SymbolicNeuralNetwork`](@ref).

# Constructors

    Gradient(f, nn)

Differentiate the symbolic `f` with respect to the parameters of `nn`.

    Gradient(nn)

Compute the symbolic output of `nn` and differentiate it with respect to the parameters of `nn`. This does:

```julia
nn.model(nn.input, params(nn))
```

# Examples

```julia
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
(Gradient(nn) |> derivative)[1].L1.b
```

# Implementation

Internally the constructors are using [`symbolic_pullback`](@ref).
"""
struct Gradient{OT, SDT, ST} <: Derivative{OT, SDT, ST} 
    f::OT
    ∇::SDT
    nn::ST
end

"""
    derivative(g)

# Examples

We compare this to [`symbolic_pullback`](@ref) here:

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative, symbolic_pullback
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
g = Gradient(nn)
∇ = derivative(g)

isequal(∇, symbolic_pullback(g.f, nn))

# output

true
```
"""
derivative(g::Gradient) = g.∇

function Gradient(output::EqT, nn::SymbolicNeuralNetwork)
    typeof(output) <: AbstractArray ? nothing : (@warn "You should only use `Gradient` together with array expressions! Maybe you wanted to use `SymbolicPullback`.")
    Gradient(output, symbolic_pullback(output, nn), nn)
end

function Gradient(nn::SymbolicNeuralNetwork)
    Gradient(nn.model(nn.input, params(nn)), nn)
end

@doc raw"""
    symbolic_pullback(f, nn)

This takes a symbolic `f`` that depends on the parameters in `nn` and returns the corresponding pullback (a symbolic expression).

This is used by [`Gradient`](@ref) and [`SymbolicPullback`](@ref).

# Examples

```julia
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, symbolic_pullback
using AbstractNeuralNetworks
using AbstractNeuralNetworks: params
using LinearAlgebra: norm

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
output = c(nn.input, params(nn))
spb = symbolic_pullback(output, nn)

spb[1].L1.b
```
"""
function symbolic_pullback(f::EqT, nn::AbstractSymbolicNeuralNetwork)::Union{AbstractArray{<:Union{NamedTuple, NeuralNetworkParameters}}, Union{NamedTuple, NeuralNetworkParameters}}
    symbolic_diffs = symbolic_differentials(params(nn))
    [symbolic_derivative(f_single, symbolic_diffs) for f_single ∈ f]
end