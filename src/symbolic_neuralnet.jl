
abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

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

    # Generation of symbolicparamters
    sparams = symbolicparameters(model)

    # Generation of the equations

    eval = model(sinput, sparams)

    infos = merge(NamedTuple{keys(new_eqs)}(Tuple(typeof(eq) <: Vector{<:Real} ? 1 : 2 for eq in new_eqs)),(eval = typeof(eval) <: Vector{<:Real} ? 1 : 2,))

    pre_equations = Tuple(SymbolicUtils.substitute.(eq, [snn => eval]) for eq in new_eqs)

    pre_equations = Tuple(Symbolics.scalarize.(eq) for eq in pre_equations)

    pre_equations = Tuple(expand_derivatives.(eq) for eq in pre_equations)

    equations = NamedTuple{keys(new_eqs)}(pre_equations)

    # Gestion of evaluation function

    pre_code_eval = build_function(eval, sinput, sparams...)

    code_eval = typeof(pre_code_eval) <: Tuple ? pre_code_eval[infos[:eval]] : pre_code_eval

    rewrite_code_eval = rewrite_neuralnetwork(code_eval, (sinput,), sparams)

    code_opti_eval = optimize_code!(rewrite_code_eval)

    code_corr_eval = Meta.parse(replace(string(code_opti_eval), "SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), Val{(2,)}()," => "(" ))
    
    # Generation of the code for equations

    pre_code = Tuple((build_function(eq, sinput, sparams...), infos[keq]) for (keq,eq) in pairs(equations))

    code = Tuple(typeof(c) <: Tuple ? c[i] : c for (c,i) in pre_code) 

    # Rewrite of the codes
    rewrite_codes = Tuple(rewrite_neuralnetwork(c, (sinput,), sparams) for c in code)

    # Optimization of the code

    #code_opti = optimize_code!.(rewrite_codes)

    #code_corr = Meta.parse.(replace.(string.(code_opti), "SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), Val{(2,)}()," => "(" ))
    
    # Creation of NamedTuple code
    nt_code = merge(NamedTuple{keys(equations)}(rewrite_codes),(eval = code_corr_eval,))

    # Generations of the functions
    functions = NamedTuple{keys(nt_code)}(Tuple(@RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(c)) for c in nt_code))

    SymbolicNeuralNetwork(arch, model, sparams, nt_code, functions.eval, merge(equations, (eval = eval,)), functions)
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


