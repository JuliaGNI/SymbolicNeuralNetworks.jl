abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

"""
    SymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A symbolic neural network realizes a symbolic represenation (of small neural networks).

The `struct` has the following fields:
- `architecture`: the neural network architecture,
- `model`: the model (typically a Chain that is the realization of the architecture),
- `params`: the symbolic parameters of the network,
- `equations`: gives a latex representation of the equations in terms of the neural network parameters (uses `latexify`),
- `functions`: same as in the field `equations`, but in form of executable Julia code.

# Constructors

    SymbolicNeuralNetwork(arch; eqs)

Make a `SymbolicNeuralNetwork` based on an architecture and a set of equations.
`eqs` here has to be a `NamedTuple` that contains keys 
- `:x`: gives the inputs to the neural network and 
- `:nn`: symbolic expression of the neural network.

Internally this calls [`evaluate_equations`](@ref)
"""
struct SymbolicNeuralNetwork{AT, MT, PT, ET, FT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT

    equations::ET
    functions::FT
end

@inline architecture(snn::SymbolicNeuralNetwork) = snn.architecture
@inline model(snn::SymbolicNeuralNetwork) = snn.model
@inline params(snn::SymbolicNeuralNetwork) = snn.params

@inline equations(snn::SymbolicNeuralNetwork) = snn.equations
@inline functions(snn::SymbolicNeuralNetwork) = snn.functions

function SymbolicNeuralNetwork(arch::Architecture, model::Model; eqs::NamedTuple)

    @assert [:x, :nn] ⊆ keys(eqs)
    
    sinput = eqs.x

    snn = eqs.nn

    s∇nn = haskey(eqs, :∇nn) ? eqs.∇nn : nothing

    sJnn = haskey(eqs, :Jnn) ? eqs.Jnn : nothing

    remaining_eqs = NamedTuple([p for p in pairs(eqs) if p[1] ∉ [:x, :nn, :∇nn, :Jnn]])

    # Generation of symbolic paramters
    sparams = symbolicparameters(model)

    # Evaluation of the symbolic output
    soutput = model(sinput, sparams)

    # Evaluation of gradient
    s∇output = isnothing(s∇nn) ? nothing : Symbolics.gradient(sum(Symbolics.scalarize(soutput)), sinput)

    # Evaluation of the Jacobian matrix
    sJoutput = isnothing(sJnn) ? nothing : Symbolics.jacobian(soutput, sinput)

    equations = evaluate_equations(remaining_eqs, snn, s∇nn, sJnn, soutput, s∇output, sJoutput)

    # Generations of the functions
    functions = _generate_functions(equations, sinput, sparams)

    equations_latex = NamedTuple{keys(equations)}(Tuple(latexify(eq) for eq in equations))
    SymbolicNeuralNetwork(arch, model, sparams, equations_latex, functions)
end

"""
    substitute_jacobian(eq, sJnn, sJoutput)

Substitute the symbolic expression `sJnn` in `eq` with the symbolic expression `sJoutput`.

# Implementation 

See the comment in [`evaluate_equation`](@ref).
"""
function substitute_jacobian(eq, sJnn, sJoutput)
    @assert axes(sJnn) == axes(sJoutput)
    substitute.(eq, Ref(Dict([sJnn[i, j] => sJoutput[i, j] for (i, j) in axes(sJnn)])))
end

function substitute_jacobian(eq, ::Nothing, ::Nothing)
    eq
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
This is also the case for the functions [`substitute_gradient`](@ref) and [`substitute_jacobian`](@ref).
"""
function evaluate_equation(eq, snn, s∇nn, sJnn, soutput, s∇output, sJoutput)
    @assert axes(snn) == axes(soutput)
    eq_output_substituted = substitute.(eq, Ref(Dict([snn[i] => soutput[i] for i in axes(snn, 1)])))
    eq_gradient_substituted = substitute_gradient(eq_output_substituted, s∇nn, s∇output)
    eq_jacobian_substituted = substitute_jacobian(eq_gradient_substituted, sJnn, sJoutput)
    simplify(eq_jacobian_substituted)
end

"""
    evaluate_equations(eqs, soutput)

Apply [`evaluate_equation`](@ref) to a `NamedTuple` and append `(soutput = soutput, s∇output = s∇output, sJoutput = sJoutput)`.
"""
function evaluate_equations(eqs::NamedTuple, snn, s∇nn, sJnn, soutput, s∇output, sJoutput)
    
    # closure
    _evaluate_equation(eq) = evaluate_equation(eq, snn, s∇nn, sJnn, soutput, s∇output, sJoutput)
    evaluated_equations = Tuple(_evaluate_equation(eq) for eq in eqs)

    soutput_eq = (soutput = simplify(soutput),)
    s∇output_eq = isnothing(s∇nn) ? NamedTuple() : (s∇output = simplify(s∇output),)
    sJoutput_eq = isnothing(sJnn) ? NamedTuple() : (sJoutput = simplify(sJoutput),)
    merge(NamedTuple{keys(eqs)}(evaluated_equations), soutput_eq, s∇output_eq, sJoutput_eq)
end

function _generate_function(expression, sinput, sparams)
    # we use the second element as the first one is the inplace version of the function
    build_function(expression, sinput, sparams; expression=Val{false})
end

function _generate_functions(expressions::NamedTuple, sinput, sparams)
    NamedTuple{keys(expressions)}(Tuple(_generate_function(expression, sinput, sparams) for expression in expressions))
end

function SymbolicNeuralNetwork(model::Model; kwargs...)
    SymbolicNeuralNetwork(UnknownArchitecture(), model; kwargs...)
end

function SymbolicNeuralNetwork(arch::Architecture; kwargs...)
    SymbolicNeuralNetwork(arch, Chain(arch); kwargs...)
end

function SymbolicNeuralNetwork(arch::Architecture, model::Model, dim::Int)
    @variables nn
    x = Symbolics.variables(:x, 1:dim)
    SymbolicNeuralNetwork(arch, model; eqs = (x = x, nn = nn))
end

function SymbolicNeuralNetwork(arch::Architecture, dim::Int)
    SymbolicNeuralNetwork(arch, Chain(arch), dim)
end

function SymbolicNeuralNetwork(model::Model, dim::Int)
    SymbolicNeuralNetwork(UnknownArchitecture(), model, dim)
end

function SymbolicNeuralNetwork(d::AbstractExplicitLayer{M}) where M
    SymbolicNeuralNetwork(d, M)
end

(snn::SymbolicNeuralNetwork)(x, params) = snn.functions.soutput(x, params)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)


function Base.show(io::IO, snn::SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
    print(io, "\nArchitecture = ")
    print(io, architecture(snn))
    print(io, "\nModel = ")
    print(io, model(snn))
    print(io, "\nSymbolic Params = ")
    print(io, params(snn))
    print(io, "\n\nand equations of motion\n\n")
    for eq in equations(snn)
        print(io, eq)
        print(io, "\n")
    end
end