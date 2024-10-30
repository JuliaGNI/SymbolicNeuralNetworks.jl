using Test, SymbolicNeuralNetworks
using AbstractNeuralNetworks: Dense, initialparameters
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
"""
function test_gradient(n::Integer, T = Float32)
    d = Dense(n, 1, tanh)
    Symbolics.@variables nn ∇nn
    x = Symbolics.variables(:x, 1:n)
    eqs = (x = x, nn = nn, ∇nn = ∇nn)
    nn = SymbolicNeuralNetwork(d; eqs = eqs)

    params = initialparameters(d, T)
    input = rand(n)
    @test nn.functions.soutput[2](input, params) ≈ d(input, params)
    @test nn.functions.s∇output[1](input, params) ≈ ForwardDiff.gradient(input -> sum(d(input, params)), input)
end

for n ∈ 1:10
    for T ∈ (Float32, Float64)
        test_gradient(n, T)
    end
end