@doc raw"""
    Jacobian <: Derivative

Computes and stores the derivative of a symbolic expression with respect to the *input* of a
[`SymbolicNeuralNetwork`](@ref).

# Constructors

    Jacobian(f, nn)
    Jacobian(nn)

Differentiate the symbolic `f` with respect to the input of `nn`. If `f` is not supplied it is taken
to be the symbolic output of the network, `nn.model(nn.input, params(nn))`.

# Fields

1. `f`: the symbolic expression that was differentiated,
2. `□`: the symbolic Jacobian,
3. `nn`: the [`SymbolicNeuralNetwork`](@ref).

# Implementation

For a function ``f:\mathbb{R}^n\to\mathbb{R}^m`` we use the convention

```math
\square_{ij} = \frac{\partial}{\partial{}x_j}f_i, \text{ i.e. } \square \in \mathbb{R}^{m\times{}n},
```

which is also the one [`Zygote`](https://github.com/FluxML/Zygote.jl) and
[`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) use. An `f` that is not a vector is
flattened with `vec` first, so the rows of `□` are indexed by `vec(f)`; a *scalar* `f` gives a
``1\times{}n`` Jacobian, i.e. its gradient with respect to the input as a row.

# Examples

Here we compute the Jacobian of a single-layer neural network ``x \mapsto \mathrm{tanh}(Wx + b)``,
whose element-wise derivative is

```math
    \frac{\partial}{\partial{}x_i}\sigma\left(\sum_{k}w_{jk}x_k + b_j\right) = \sigma'\left(\sum_{k}w_{jk}x_k + b_j\right)w_{ji},
```

and compare it to that expression. Note that ``\mathrm{tanh}'(x) = \frac{4e^{2x}}{(e^{2x} + 1)^2}.``

```jldoctest
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative
using AbstractNeuralNetworks: Dense, Chain, NeuralNetwork, params
import Random

Random.seed!(123)

input_dim = 5
output_dim = 2
c = Chain(Dense(input_dim, output_dim, tanh))
nn = SymbolicNeuralNetwork(c)
jacobian = build_nn_function(derivative(Jacobian(nn)), nn)

ps = params(NeuralNetwork(c, Float64))
input = rand(input_dim)
Dtanh(x::Real) = 4 * exp(2 * x) / (1 + exp(2x)) ^ 2
analytic_jacobian(i, j) = Dtanh(sum(k -> ps.L1.W[j, k] * input[k], 1:input_dim) + ps.L1.b[j]) * ps.L1.W[j, i]
jacobian(input, ps) ≈ [analytic_jacobian(i, j) for j ∈ 1:output_dim, i ∈ 1:input_dim]

# output

true
```
"""
struct Jacobian{OT, SDT, ST} <: Derivative{OT, SDT, ST}
    f::OT
    □::SDT
    nn::ST
end

function Jacobian(f, nn::AbstractSymbolicNeuralNetwork)
    differentials = symbolic_differentials(nn.input)
    rows = _flat_entries(scalar_expressions(f))
    □ = [expand_derivatives(D(row)) for row in rows, D in differentials]
    Jacobian(f, □, nn)
end

Jacobian(nn::AbstractSymbolicNeuralNetwork) = Jacobian(nn.model(nn.input, params(nn)), nn)

"""
    derivative(j)

The symbolic Jacobian stored in `j`.
"""
derivative(j::Jacobian) = j.□
