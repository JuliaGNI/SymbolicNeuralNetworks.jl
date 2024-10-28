abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

"""
    SymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A symbolic neural network realizes a symbolic represenation (of small neural networks).

The `struct` has the following fields:
- `architecture`: the neural network architecture,
- `model`: the model (typically a Chain that is the realization of the architecture),
- `params`: the symbolic parameters of the network,
- `code`:
- `eval`:
- `equations`:
- `functions`: 

# Constructors

    SymbolicNeuralNetwork(arch; eqs)

Make a `SymbolicNeuralNetwork` based on an architecture and a set of equations.
`eqs` here has to be a `NamedTuple` that contains keys 
- `:x`: gives the inputs to the neural network and 
- `:nn`: symbolic expression of the neural network.

Internally this calls [`evaluate_equations`](@ref)
"""
struct SymbolicNeuralNetwork{AT,MT,PT,CT,EVT,ET,FT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT

    code::CT
    eval::EVT
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
    
    RuntimeGeneratedFunctions.init(@__MODULE__)

    sinput = eqs.x
    snn = eqs.nn

    new_eqs = NamedTuple([p for p in pairs(eqs) if p[1] ∉ [:x, :nn]])

    # Generation of symbolic paramters
    sparams = symbolicparameters(model)

    # Evaluation of the symbolic input
    eval = model(sinput, sparams)

    infos = merge(classify_equations(new_eqs), (eval = classify_equation(eval),))

    equations = evaluate_equations(new_eqs, eval)

    # Generation of the code
    codes = generate_codes(equations, sinput, sparams, infos)

    # Generations of the functions
    functions = generate_functions(codes)

    SymbolicNeuralNetwork(arch, model, sparams, codes, functions.eval, equations, functions)
end

# return "1" if input is vector of reals; else return "2"
function classify_equation(eq)::Integer
    typeof(eq) <: Vector{<:Real} ? 1 : 2
end

function classify_equations(eqs::NamedTuple)
    NamedTuple{keys(eqs)}(Tuple(classify_equation(eq) for eq in eqs))
end

"""
    evaluate_equation(eq, eval)

Replace `snn` in `eq` with `eval` (input), scalarize and expand derivatives.

This uses `scalarize` and `expand_derivatives` from the `Symbolics` package.
"""
function evaluate_equation(eq, eval)
    SymbolicUtils.substitute(eq, [snn => eval]) |> Symbolics.scalarize |> expand_derivatives
end

"""
    evaluate_equations(eqs, eval)

Apply [`evaulate_equation`](@ref) to a `NamedTuple` and append the `NamedTuple` `(eval = eval, )`.
"""
function evaluate_equations(eqs::NamedTuple, eval)
    
    pre_equations = Tuple(evaluate_equation.(eq) for eq in eqs)

    merge(NamedTuple{keys(eqs)}(pre_equations),(eval = eval,))
end

"""
    generate_code(eq, sinput, sparams, info)

Generate code according to the equation `eq`, symbolic input `sinput`, symbolic neural network parameters `sparams` and `info`.

# Arguments

The last argument, `info`, is an integer that's either `1` or `2`. This is the output of the function `classify_equation`.
"""
function generate_code(eq, sinput, sparams, info)
    @assert info ∈ [1, 2]
    pre_code = build_function(eq, sinput, sparams...)
    code = typeof(pre_code) <: Tuple ? pre_code[info] : pre_code
    code_rewritten = rewrite_neuralnetwork(code, (sinput, ), sparams)
    code_opti = optimize_code!(code_rewritten)
    Meta.parse(replace(string(code_opti), "SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), Val{(2,)}()," => "(" ))
end

function generate_codes(eqs::NamedTuple, sinput, sparams, infos::NamedTuple)
    @assert keys(eqs) == keys(infos)
    generated_code_tuples = Tuple(generate_code(eqs[key], sinput, sparams, infos[key]) for key in keys(eqs))
    NamedTuple{keys(eqs)}(generated_code_tuples)
end

function generate_function(c)
    @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(c))
end

function generate_functions(codes::NamedTuple)
    NamedTuple{keys(codes)}(Tuple(generate_function(c) for c in codes))
end

function SymbolicNeuralNetwork(model::Model; kwargs...)
    SymbolicNeuralNetwork(UnknownArchitecture(), model; kwargs...)
end

function SymbolicNeuralNetwork(arch::Architecture; kwargs...)
    SymbolicNeuralNetwork(arch, Chain(arch); kwargs...)
end

function SymbolicNeuralNetwork(arch::Architecture, model::Model, dim::Int)
    @variables x[1:dim], nn
    SymbolicNeuralNetwork(arch, model; eqs = (x = x, nn = nn))
end

function SymbolicNeuralNetwork(arch::Architecture, dim::Int)
    SymbolicNeuralNetwork(arch, Chain(arch), dim)
end

function SymbolicNeuralNetwork(model::Model, dim::Int)
    SymbolicNeuralNetwork(UnknownArchitecture(), model, dim)
end

(snn::SymbolicNeuralNetwork)(x, params) = snn.functions.eval(x, params)
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


