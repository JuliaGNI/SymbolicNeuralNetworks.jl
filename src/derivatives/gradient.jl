@doc raw"""
    Gradient <: Derivative

Computes and stores the gradient of a symbolic function with respect to the parameters of a [`SymbolicNeuralNetwork`](@ref).

# Constructors

    Gradient(output, nn)

Differentiate the symbolic `output` with respect to the parameters of `nn`.

    Gradient(nn)

Compute the symbolic output of `nn` and differentiate it with respect to the parameters of `nn`.

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative
using AbstractNeuralNetworks
using Latexify: latexify

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
(Gradient(nn) |> derivative)[1].L1.b |> latexify

# output

L"\begin{equation}
\left[
\begin{array}{c}
1 - \tanh^{2}\left( \mathtt{W\_2}_{1} + \mathtt{W\_1}_{1,1} \mathtt{sinput}_{1} + \mathtt{W\_1}_{1,2} \mathtt{sinput}_{2} \right) \\
\end{array}
\right]
\end{equation}
"
```

# Implementation

Internally the constructors are using [`symbolic_pullback`](@ref).
"""
struct Gradient{ST, OT, SDT} <: Derivative{ST, OT, SDT} 
    nn::ST
    output::OT
    ∇::SDT
end

"""
    derivative(g)

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative, symbolic_pullback
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
g = Gradient(nn)
∇ = derivative(g)

isequal(∇, symbolic_pullback(g.output, nn))

# output

true
```
"""
derivative(g::Gradient) = g.∇

function Gradient(output::EqT, nn::SymbolicNeuralNetwork)
    typeof(output) <: AbstractArray ? nothing : (@warn "You should only use `Gradient` together with array expressions! Maybe you wanted to use `SymbolicPullback`.")
    Gradient(nn, output, symbolic_pullback(output, nn))
end

function Gradient(nn::SymbolicNeuralNetwork)
    Gradient(nn.model(nn.input, params(nn)), nn)
end

@doc raw"""
    symbolic_pullback(nn, output)

This takes a symbolic output that depends on the parameters in `nn` and returns the corresponding pullback (a symbolic expression).

This is used by [`Gradient`](@ref) and [`SymbolicPullback`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, symbolic_pullback
using AbstractNeuralNetworks
using AbstractNeuralNetworks: params
using LinearAlgebra: norm
using Latexify: latexify

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
output = c(nn.input, params(nn))
spb = symbolic_pullback(output, nn)

spb[1].L1.b |> latexify

# output

L"\begin{equation}
\left[
\begin{array}{c}
1 - \tanh^{2}\left( \mathtt{W\_2}_{1} + \mathtt{W\_1}_{1,1} \mathtt{sinput}_{1} + \mathtt{W\_1}_{1,2} \mathtt{sinput}_{2} \right) \\
\end{array}
\right]
\end{equation}
"
```
"""
function symbolic_pullback(soutput::EqT, nn::AbstractSymbolicNeuralNetwork)::Union{AbstractArray{<:Union{NamedTuple, NeuralNetworkParameters}}, Union{NamedTuple, NeuralNetworkParameters}}
    symbolic_diffs = symbolic_differentials(params(nn))
    [symbolic_derivative(soutput_single, symbolic_diffs) for soutput_single ∈ soutput]
end