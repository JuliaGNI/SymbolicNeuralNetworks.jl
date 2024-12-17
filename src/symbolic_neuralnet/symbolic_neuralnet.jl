abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

"""
    SymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A symbolic neural network realizes a symbolic represenation (of small neural networks).

# Fields

The `struct` has the following fields:
- `architecture`: the neural network architecture,
- `model`: the model (typically a Chain that is the realization of the architecture),
- `params`: the symbolic parameters of the network.
- `sinput`: the symbolic input of the network.

# Constructors

    SymbolicNeuralNetwork(nn)

Make a `SymbolicNeuralNetwork` based on a `AbstractNeuralNetworks.Network`.
"""
struct SymbolicNeuralNetwork{   AT, 
                                MT, 
                                PT <: Union{NeuralNetworkParameters, NamedTuple}, 
                                IT  } <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
    input::IT
end

function SymbolicNeuralNetwork(nn::NeuralNetwork)
    cache = Dict()
    sparams = symbolize!(cache, params(nn), :W)
    @variables sinput[1:input_dimension(nn.model)]

    SymbolicNeuralNetwork(nn.architecture, nn.model, sparams, sinput)
end

function SymbolicNeuralNetwork(arch::Architecture, model::Model)
    nn = NeuralNetwork(arch, model, CPU(), Float64)
    SymbolicNeuralNetwork(nn)
end

function SymbolicNeuralNetwork(model::Chain)
    SymbolicNeuralNetwork(UnknownArchitecture(), model)
end

function SymbolicNeuralNetwork(arch::Architecture)
    SymbolicNeuralNetwork(arch, Chain(arch))
end

function SymbolicNeuralNetwork(d::AbstractExplicitLayer)
    SymbolicNeuralNetwork(UnknownArchitecture(), d)
end

params(snn::AbstractSymbolicNeuralNetwork) = snn.params

apply(snn::AbstractSymbolicNeuralNetwork, x, args...) = snn(x, args...)

input_dimension(::AbstractExplicitLayer{M}) where M = M 
input_dimension(c::Chain) = input_dimension(c.layers[begin])
output_dimension(::AbstractExplicitLayer{M, N}) where {M, N} = N
output_dimension(c::Chain) = output_dimension(c.layers[end])

function Base.show(io::IO, snn::SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
    print(io, "\nArchitecture = ")
    print(io, snn.architecture)
    print(io, "\nModel = ")
    print(io, snn.model)
    print(io, "\nSymbolic Params = ")
    print(io, params(snn))
end