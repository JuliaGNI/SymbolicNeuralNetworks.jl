"""
    symbolicparameters(model)

Obtain the symbolic parameters of a neural network model.

# Examples

```jldoctest
using SymbolicNeuralNetworks: symbolize!
using AbstractNeuralNetworks

cache = Dict()
d = Dense(4, 5, tanh)
params = NeuralNetwork(Chain(d)).params.L1
symbolize!(cache, params, :X) |> typeof

# output

@NamedTuple{W::Symbolics.Arr{Symbolics.Num, 2}, b::Symbolics.Arr{Symbolics.Num, 1}}
```
"""
symbolicparameters(model::Model) = error("symbolicparameters not implemented for model type ", typeof(model))