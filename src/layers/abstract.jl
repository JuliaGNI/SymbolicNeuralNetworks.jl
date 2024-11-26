"""
    symbolicparameters(model)

Obtain the symbolic parameters of a neural network model.

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks

d = Dense(4, 5, tanh)
symbolicparameters(d) |> typeof

# output

@NamedTuple{W::Symbolics.Arr{Symbolics.Num, 2}, b::Symbolics.Arr{Symbolics.Num, 1}}
```
"""
symbolicparameters(model::Model) = error("symbolicparameters not implemented for model type ", typeof(model))