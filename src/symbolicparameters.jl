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

function symbolicparameters(model::Chain)
    symbolize(Tuple(symbolicparameters(layer) for layer in model))[1]
end


function symbolicparameters(::Dense{M,N,true}) where {M,N}
    @variables W[1:N, 1:M], b[1:N]
    (W = W, b = b)
end

function symbolicparameters(::Dense{M,N,false}) where {M,N}
    @variables W[1:N, 1:M]
    (W = W,)
end

