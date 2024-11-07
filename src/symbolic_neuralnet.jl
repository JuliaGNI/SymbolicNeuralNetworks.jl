abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

# define custom equation type
const EqT = Union{Symbolics.Arr{Num}, AbstractArray{Num}, Num}

"""
    SymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A symbolic neural network realizes a symbolic represenation (of small neural networks).

The `struct` has the following fields:
- `architecture`: the neural network architecture,
- `model`: the model (typically a Chain that is the realization of the architecture),
- `params`: the symbolic parameters of the network.

# Constructors

    SymbolicNeuralNetwork(arch)

Make a `SymbolicNeuralNetwork` based on an architecture and a set of equations.
"""
struct SymbolicNeuralNetwork{AT, MT, PT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
end

function SymbolicNeuralNetwork(arch::Architecture, model::Model)
    # Generation of symbolic paramters
    sparams = symbolicparameters(model)

    SymbolicNeuralNetwork(arch, model, sparams)
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
    merge(NamedTuple{keys(eqs)}(evaluated_equations), soutput_eq, s∇output_eq)
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
function evaluate_equations(eqs::NamedTuple, nn::SymbolicNeuralNetwork; kwargs...)
    @assert [:x, :nn] ⊆ keys(eqs)
    
    sinput = eqs.x

    snn = eqs.nn

    s∇nn = haskey(eqs, :∇nn) ? eqs.∇nn : nothing

    remaining_eqs = NamedTuple([p for p in pairs(eqs) if p[1] ∉ [:x, :nn, :∇nn]])

    # Evaluation of the symbolic output
    soutput = _scalarize(nn.model(sinput, nn.params))

    # make differential 
    Dx = collect(Differential.(sinput))

    # Evaluation of gradient
    s∇output = isnothing(s∇nn) ? nothing :  [dx(soutput) for dx in Dx]

    evaluate_equations(remaining_eqs, snn, s∇nn, soutput, s∇output; kwargs...)
end

function SymbolicNeuralNetwork(model::Chain)
    SymbolicNeuralNetwork(UnknownArchitecture(), model)
end

function SymbolicNeuralNetwork(arch::Architecture)
    SymbolicNeuralNetwork(arch, Chain(arch))
end

function SymbolicNeuralNetwork(d::AbstractExplicitLayer{M}) where M
    SymbolicNeuralNetwork(d, M)
end

(snn::SymbolicNeuralNetwork)(x, params) = snn.functions.soutput(x, params)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)

function _scalarize(y::Symbolics.Arr{Num})
    @assert axes(y) == (1:1,)
    sum(y)
end

"""
    gradient(nn)

Compute the gradient of a [`SymbolicNeuralNetwork`](@ref) with respect to the input arguments.

The output of `gradient` consists of
1. a symbolic expression of the gradient,
2. a symbolic expression of the input.
"""
function gradient(nn::SymbolicNeuralNetwork)
    x, output, ∇nn = @variables x[1:2] output[1:1] ∇nn[1:2]
    eqs = (x = x, nn = output, ∇nn = ∇nn)
    s∇output = evaluate_equations(eqs, nn)
    s∇output, x
end

function Base.show(io::IO, snn::SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
    print(io, "\nArchitecture = ")
    print(io, snn.architecture)
    print(io, "\nModel = ")
    print(io, snn.model)
    print(io, "\nSymbolic Params = ")
    print(io, snn.params)
end