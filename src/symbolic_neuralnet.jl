abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

# define custom equation type
const EqT = Union{Symbolics.Arr{Num}, AbstractArray{Num}, Num, AbstractArray{<:Symbolics.BasicSymbolic}}

"""
    SymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A symbolic neural network realizes a symbolic represenation (of small neural networks).

The `struct` has the following fields:
- `architecture`: the neural network architecture,
- `model`: the model (typically a Chain that is the realization of the architecture),
- `params`: the symbolic parameters of the network.
- `sinput`: the symbolic input of the network.

# Constructors

    SymbolicNeuralNetwork(arch)

Make a `SymbolicNeuralNetwork` based on an architecture and a set of equations.
"""
struct SymbolicNeuralNetwork{AT, MT, PT <: NeuralNetworkParameters, IT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
    input::IT
end

function SymbolicNeuralNetwork(arch::Architecture, model::Model)
    # Generation of symbolic paramters
    sparams = symbolicparameters(model)
    @variables sinput[1:input_dimension(model)]

    SymbolicNeuralNetwork(arch, model, sparams, sinput)
end

"""
    substitute_gradient(eq, s∇nn, s∇output)

Substitute the symbolic expression `s∇nn` in `eq` with the symbolic expression `s∇output`.

# Implementation 

See the comment in [`evaluate_equation`](@ref).
"""
function substitute_gradient(eq, s∇nn, s∇output)
    @assert axes(s∇nn) == axes(s∇output)
    substitute.(eq, Ref(Dict([s∇nn[i] => s∇output[i] for i in axes(s∇nn, 1)])))
end

function substitute_gradient(eq, ::Nothing, ::Nothing)
    eq 
end

"""
    evaluate_equation(eq, soutput)

Replace `snn` in `eq` with `soutput` (input), scalarize and expand derivatives.

# Implementation

Here we use `Symbolics.substitute` with broadcasting to be able to handle `eq`s that are arrays.
For that reason we use [`Ref` before `Dict`](https://discourse.julialang.org/t/symbolics-and-substitution-using-broadcasting/68705).
This is also the case for the functions [`substitute_gradient`](@ref).
"""
function evaluate_equation(eq::EqT, snn::EqT, s∇nn::EqT, soutput::EqT, s∇output::EqT)
    @assert axes(snn) == axes(soutput)
    eq_output_substituted = substitute.(eq, Ref(Dict([snn[i] => soutput[i] for i in axes(snn, 1)])))
    substitute_gradient(eq_output_substituted, s∇nn, s∇output)
end

"""
    evaluate_equations(eqs, soutput)

Apply [`evaluate_equation`](@ref) to a `NamedTuple` and append `(soutput = soutput, s∇output = s∇output)`.
"""
function evaluate_equations(eqs::NamedTuple, snn::EqT, s∇nn::EqT, soutput::EqT, s∇output::EqT; simplify = true)
    
    # closure
    _evaluate_equation(eq) = evaluate_equation(eq, snn, s∇nn, soutput, s∇output)
    evaluated_equations = Tuple(_evaluate_equation(eq) for eq in eqs)

    soutput_eq = (soutput = simplify == true ? Symbolics.simplify(soutput) : soutput,)
    s∇output_eq = isnothing(s∇nn) ? NamedTuple() : (s∇output = simplify == true ? Symbolics.simplify(s∇output) : s∇output,)
    merge(soutput_eq, s∇output_eq, NamedTuple{keys(eqs)}(evaluated_equations))
end

"""
    expand_parameters(eqs, nn)

Expand the output and gradient in `eqs` with the weights in `nn`.

`eqs` here has to be a `NamedTuple` that contains keys 
- `:x`: gives the inputs to the neural network and 
- `:nn`: symbolic expression of the neural network.

# Implementation

Internally this 
1. computes the gradient and
2. calls [`evaluate_equations(::NamedTuple, ::EqT, ::EqT, ::EqT, EqT)`](@ref).

"""
function evaluate_equations(eqs::NamedTuple, nn::AbstractSymbolicNeuralNetwork; kwargs...)
    @assert [:x, :nn] ⊆ keys(eqs)
    
    sinput = eqs.x

    snn = eqs.nn

    s∇nn = haskey(eqs, :∇nn) ? eqs.∇nn : nothing

    remaining_eqs = NamedTuple([p for p in pairs(eqs) if p[1] ∉ [:x, :nn, :∇nn]])

    # Evaluation of the symbolic output
    soutput = _scalarize(nn.model(sinput, nn.params))

    # make differential 
    Dx = symbolic_differentials(sinput)

    # Evaluation of gradient
    s∇output = isnothing(s∇nn) ? nothing : [expand_derivatives(Symbolics.scalarize(dx(soutput))) for dx in Dx]

    evaluate_equations(remaining_eqs, snn, s∇nn, soutput, s∇output; kwargs...)
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

apply(snn::AbstractSymbolicNeuralNetwork, x, args...) = snn(x, args...)

function _scalarize(y::Symbolics.Arr{Num})
    @assert axes(y) == (1:1,)
    sum(y)
end

"""
    gradient(nn)

Compute the gradient of a [`SymbolicNeuralNetwork`](@ref) with respect to the input arguments.

The output of `gradient` consists of a `NamedTuple` that has the following keys:
1. a symbolic expression of the input (keyword `x`),
2. a symbolic expression of the output (keyword `soutput`),
3. a symbolic expression of the gradient (keyword `s∇output`).

# Examples

Using `gradient` is equivalent to calling the more complicated function [`evaluate_equations`](@ref) with the appropriate arguments:

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Dense
using Symbolics
import Latexify

input_dim = 5
d = Dense(input_dim, 1, tanh)
nn = SymbolicNeuralNetwork(d)
@variables x[1:input_dim] ∇nn[1:input_dim] output[1:1]
eqs = (x = x, nn = output, ∇nn = ∇nn)
evaluated_equations = evaluate_equations(eqs, nn)
evaluated_equations2 = SymbolicNeuralNetworks.gradient(nn)
Latexify.latexify(evaluated_equations.s∇output) == Latexify.latexify(evaluated_equations2.s∇output)

# output

true
```
"""
function gradient(nn::AbstractSymbolicNeuralNetwork)
    @assert output_dimension(nn.model) == 1 "Output dimension has to be 1 to be able to compute the gradient."
    input_dim = input_dimension(nn.model)
    x, output, ∇nn = @variables x[1:input_dim] output[1:1] ∇nn[1:input_dim]
    eqs = (x = x, nn = output, ∇nn = ∇nn)
    evaluated_equations = evaluate_equations(eqs, nn)
    merge((x = x,), evaluated_equations)
end

input_dimension(::AbstractExplicitLayer{M}) where M = M 
input_dimension(c::Chain) = input_dimension(c.layers[1])
output_dimension(::AbstractExplicitLayer{M, N}) where {M, N} = N
output_dimension(c::Chain) = output_dimension(c.layers[end])

function Base.show(io::IO, snn::SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
    print(io, "\nArchitecture = ")
    print(io, snn.architecture)
    print(io, "\nModel = ")
    print(io, snn.model)
    print(io, "\nSymbolic Params = ")
    print(io, snn.params)
end