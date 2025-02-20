using Test, SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using LinearAlgebra: norm
import Symbolics, Random, ForwardDiff

Random.seed!(123)

@doc raw"""
Here we take the gradient of an output of a single dense layer:

```math
    d: x \mapsto \mathrm{tanh}(v^Tx + b),
```
with ``x, v\in\mathbb{R}^n`` and ``b\in\mathbb{R}``.

The gradient of this expression is:
```math
    \nabla{}d: x \mapsto \mathrm{tanh}'(v^Tx + b)v.
```

Note that we use `Jacobian` here.
"""
function test_jacobian(n::Integer, T = Float32)
    c = Chain(Dense(n, 1, tanh))
    nn = SymbolicNeuralNetwork(c)
    g = Jacobian(nn)

    _params = params(NeuralNetwork(c, T))
    input = rand(T, n)
    @test build_nn_function(g.f, nn)(input, _params) ≈ c(input, _params)
    @test build_nn_function(derivative(g), nn)(input, _params) ≈ ForwardDiff.jacobian(input -> c(input, _params), input)
end

for n ∈ 10:1
    for T ∈ (Float32, Float64)
        test_jacobian(n, T)
    end 
end