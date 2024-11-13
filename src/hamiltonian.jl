"""
    HamiltonianSymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A struct that inherits properties from the abstract type `AbstractSymbolicNeuralNetwork`.

# Constructor

    HamiltonianSymbolicNeuralNetwork(model)

Make an instance of `HamiltonianSymbolicNeuralNetwork` based on a `Chain` or an `Architecture`.
This is similar to the constructor for [`SymbolicNeuralNetwork`](@ref) but also checks if the input dimension is even-dimensional and the output dimension is one.
"""
struct HamiltonianSymbolicNeuralNetwork{AT, MT, PT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
end

function HamiltonianSymbolicNeuralNetwork(arch::Architecture, model::Model)
    @assert iseven(input_dimension(model)) "Input dimension has to be an even number."
    @assert output_dimension(model) == 1 "Output dimension of network has to be scalar."

    sparams = symbolicparameters(model)
    HamiltonianSymbolicNeuralNetwork(arch, model, sparams)
end

HamiltonianSymbolicNeuralNetwork(model::Model) = HamiltonianSymbolicNeuralNetwork(UnknownArchitecture(), model)
HamiltonianSymbolicNeuralNetwork(arch::Architecture) = HamiltonianSymbolicNeuralNetwork(arch, Chain(model))

"""
    vector_field(nn::HamiltonianSymbolicNeuralNetwork)

Get the symbolic expression for the vector field belonging to the HNN `nn`.

# Implementation 

This is calling [`gradient`](@ref) and then multiplies the result with a Poisson tensor.
"""
function vector_field(nn::HamiltonianSymbolicNeuralNetwork)
    gradient_output = gradient(nn)
    sinput, soutput, ∇nn = gradient_output.x, gradient_output.soutput, gradient_output.s∇output
    input_dim = input_dimension(nn.model)
    n = input_dim ÷ 2
    # placeholder for one
    @variables o
    o_vec = repeat([o], n)
    𝕀 = Diagonal(o_vec)
    𝕆 = zero(𝕀)
    𝕁 = hcat(vcat(𝕆, -𝕀), vcat(𝕀, 𝕆))
    (x = sinput, nn = soutput, ∇nn = ∇nn, hvf = substitute(𝕁 * ∇nn, Dict(o => 1, )))
end