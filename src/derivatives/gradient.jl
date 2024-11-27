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
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, Gradient, derivative, latexify
using AbstractNeuralNetworks

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
(Gradient(nn) |> derivative)[1].L1.b |> latexify

# output

L"\begin{equation}
\left[
\begin{array}{c}
1 - \tanh^{2}\left( \mathtt{b\_1}_{1} + \mathtt{W\_1}_{1,1} \mathtt{sinput}_{1} + \mathtt{W\_1}_{1,2} \mathtt{sinput}_{2} \right) \\
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

function Gradient(output, nn::SymbolicNeuralNetwork)
    Gradient(nn, output, symbolic_pullback(output, nn))
end

function Gradient(nn::SymbolicNeuralNetwork)
    Gradient(nn.model(nn.input, nn.params), nn)
end

@doc raw"""
    symbolic_pullback(nn, output)

This takes a symbolic output that depends on the parameters in `nn` and returns the corresponding pullback (a symbolic expression).

# Examples

```jldoctest
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, symbolic_pullback, latexify
using AbstractNeuralNetworks
using LinearAlgebra: norm

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
output = c(nn.input, nn.params)
spb = symbolic_pullback(nn, output)

spb[1].L1.b |> latexify

# output

L"\begin{equation}
\left[
\begin{array}{c}
1 - \tanh^{2}\left( \mathtt{b\_1}_{1} + \mathtt{W\_1}_{1,1} \mathtt{sinput}_{1} + \mathtt{W\_1}_{1,2} \mathtt{sinput}_{2} \right) \\
\end{array}
\right]
\end{equation}
"
```
"""
function symbolic_pullback(soutput::EqT, nn::AbstractSymbolicNeuralNetwork)::Union{AbstractArray{<:NeuralNetworkParameters}, NeuralNetworkParameters}
    typeof(soutput) <: AbstractArray ? nothing : (@warn "You should only compute a pullback of array expressions!")

    symbolic_diffs = symbolic_differentials(nn.params)
    [symbolic_derivative(soutput_single, symbolic_diffs) for soutput_single ∈ soutput]
end


function symbolic_pullback(loss::NetworkLoss, nn::AbstractSymbolicNeuralNetwork)
    output_dim = output_dimension(nn.model)
    @variables soutput[1:output_dim]

    symbolic_loss = loss(nn.model, nn.params, nn.input, soutput)
    symbolic_pullback(nn, symbolic_loss)
end