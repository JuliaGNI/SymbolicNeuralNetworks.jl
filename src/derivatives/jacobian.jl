@doc raw"""
    Jacobian <: Derivative

An instance of [`Derivative`](@ref). Computes the derivatives of a neural network with respect to its inputs.

# Constructors

    Jacobian(output, nn)
    Jacobian(nn)

Compute the jacobian of a [`SymbolicNeuralNetwork`](@ref) with respect to the input arguments.

The output of `Jacobian` consists of a `NamedTuple` that has the following keys:
1. a symbolic expression of the input (keyword `x`),
2. a symbolic expression of the output (keyword `soutput`),
3. a symbolic expression of the gradient (keyword `s∇output`).

If `output` is not supplied as an input argument than it is taken to be:

```julia 
soutput = nn.model(nn.input, params(nn))
```

# Implementation

For a function ``f:\mathbb{R}^n\to\mathbb{R}^m`` we choose the following convention for the Jacobian:

```math
\square_{ij} = \frac{\partial}{\partial{}x_j}f_i, \text{ i.e. } \square \in \mathbb{R}^{m\times{}n}
```
This is also used by [`Zygote`](https://github.com/FluxML/Zygote.jl) and [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl).

# Examples

Here we compute the Jacobian of a single-layer neural network ``x \to \mathrm{tanh}(Wx + b)``. Its element-wise derivative is:

```math
    \frac{\partial}{\partial_i}\sigma(\sum_{k}w_{jk}x_k + b_j) = \sigma'(\sum_{k}w_{jk}x_k + b_j)w_{ji}.
```

Also note that for this calculation ``\mathrm{tanh}(x) = \frac{e^{2x} - 1}{e^{2x} + 1}`` and ``\mathrm{tanh}'(x) = \frac{4e^{2x}}{(e^{2x} + 1)^2}.``

We can use `Jacobian` together with [`build_nn_function`](@ref):

```jldoctest
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative
using AbstractNeuralNetworks: Dense, Chain, NeuralNetwork
using Symbolics
import Random

Random.seed!(123)

input_dim = 5
output_dim = 2
d = Dense(input_dim, 2, tanh)
c = Chain(d)
nn = SymbolicNeuralNetwork(c)
□ = SymbolicNeuralNetworks.Jacobian(nn)
# here we need to access the derivative and convert it into a function
jacobian1 = build_nn_function(derivative(□), nn)
ps = NeuralNetwork(c, Float64).params
input = rand(input_dim)
#derivative
Dtanh(x::Real) = 4 * exp(2 * x) / (1 + exp(2x)) ^ 2
analytic_jacobian(i, j) = Dtanh(sum(k -> ps.L1.W[j, k] * input[k], 1:input_dim) + ps.L1.b[j]) * ps.L1.W[j, i]
jacobian1(input, ps) ≈ [analytic_jacobian(i, j) for j ∈ 1:output_dim, i ∈ 1:input_dim]

# output

true
```
"""
struct Jacobian{ST, OT, SDT} <: Derivative{ST, OT, SDT} 
    nn::ST
    output::OT
    □::SDT
end

derivative(j::Jacobian) = j.□

function Jacobian(nn::AbstractSymbolicNeuralNetwork)
    
    # Evaluation of the symbolic output
    soutput = nn.model(nn.input, params(nn))

    Jacobian(soutput, nn)
end

function Jacobian(soutput::EqT, nn::AbstractSymbolicNeuralNetwork)
    # make differential 
    Dx = symbolic_differentials(nn.input)

    # Evaluation of gradient
    s∇output = hcat([expand_derivatives.(Symbolics.scalarize(dx(soutput))) for dx in Dx]...)

    Jacobian(nn, soutput, s∇output)
end