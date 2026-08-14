abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

"""
    SymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A symbolic representation of a (small) neural network.

It pairs a model with symbolic stand-ins for its parameters and its input, so that symbolic
expressions can be built from it:

```julia
c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
eq = c(nn.input, params(nn))
```

Those expressions can then be differentiated ([`Jacobian`](@ref), [`Gradient`](@ref),
[`SymbolicPullback`](@ref)) and turned into executable code with [`build_nn_function`](@ref).

# Fields

- `architecture`: the neural network architecture,
- `model`: the model (typically a `Chain` that realizes the architecture),
- `params`: the symbolic parameters of the network, with the same nesting as the numeric ones,
- `input`: the symbolic input of the network, a `Vector{Num}`.

# Constructors

    SymbolicNeuralNetwork(nn)
    SymbolicNeuralNetwork(arch, model)
    SymbolicNeuralNetwork(model)
    SymbolicNeuralNetwork(arch)

Build a `SymbolicNeuralNetwork` from an `AbstractNeuralNetworks.NeuralNetwork`, from an
architecture and/or a model, or from a single layer.

# Implementation

Parameters and input are built by [`symbolic_variables`](@ref), i.e. they consist of *scalar*
symbolic variables. The parameters are named `W_1`, `W_2`, … in the order in which they appear in
the parameter set, the input entries `x₁`, `x₂`, ….
"""
struct SymbolicNeuralNetwork{AT,
                             MT,
                             PT <: Union{NeuralNetworkParameters, NamedTuple},
                             IT <: AbstractVector{Num}} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
    input::IT
end

function SymbolicNeuralNetwork(nn::NeuralNetwork)
    sparams = symbolic_variables(params(nn), :W)
    sinput = Symbolics.variables(:x, 1:input_dimension(nn.model))
    SymbolicNeuralNetwork(nn.architecture, nn.model, sparams, sinput)
end

function SymbolicNeuralNetwork(arch::Architecture, model::Model)
    SymbolicNeuralNetwork(NeuralNetwork(arch, model, CPU(), Float64))
end

SymbolicNeuralNetwork(model::Chain) = SymbolicNeuralNetwork(UnknownArchitecture(), model)
SymbolicNeuralNetwork(arch::Architecture) = SymbolicNeuralNetwork(arch, Chain(arch))
# a bare layer is wrapped in a `Chain`, so that its parameters are nested the same way as those of
# any other model and the generated code does not need a special case for them
SymbolicNeuralNetwork(layer::AbstractExplicitLayer) = SymbolicNeuralNetwork(Chain(layer))

AbstractNeuralNetworks.params(nn::SymbolicNeuralNetwork) = nn.params

"""
    input_dimension(c::Chain)
    output_dimension(c::Chain)

The dimensions a `Chain` maps between, taken from its first and last layer.

`AbstractNeuralNetworks` defines both for an `AbstractLayer`; these methods extend them to a whole
`Chain`, which is what [`SymbolicNeuralNetwork`](@ref) needs to know how many symbolic input
variables to build. They belong upstream too, see
[issue #35](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/35).
"""
input_dimension(c::Chain) = input_dimension(c.layers[begin])
output_dimension(c::Chain) = output_dimension(c.layers[end])

function Base.show(io::IO, nn::SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
    print(io, "\nArchitecture = ")
    print(io, nn.architecture)
    print(io, "\nModel = ")
    print(io, nn.model)
    print(io, "\nSymbolic Params = ")
    print(io, params(nn))
end
