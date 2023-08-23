
abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

struct SymbolicNeuralNetwork{AT,MT,PT,EVT,ET,FT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT

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

    # générer les paramètres symboliques
    sparams = symbolic_params(initialparameters(CPU(), Float64, model))[1]

    # générer les équations
    eval = model(sinput, sparams)

    equations = merge(NamedTuple{keys(new_eqs)}(Tuple(expand_derivatives(SymbolicUtils.substitute(eq, [snn => eval])) for eq in new_eqs)),(eval = eval,))

    # générer les codes 

    pre_code = Tuple(build_function(eq, sinput, sparams...) for eq in equations)

    code = Tuple(typeof(c) <: Tuple ? c[2] : c for c in pre_code)

    # rewrite  les codes
    rewrite_codes = NamedTuple{keys(equations)}(Tuple(rewrite_neuralnetwork(c, (sinput,), sparams) for c in code))

    # générer les funcitons
    functions = NamedTuple{keys(rewrite_codes)}(Tuple(@RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(c)) for c in rewrite_codes))

    SymbolicNeuralNetwork(arch, model, sparams, functions.eval, equations, functions)
end

function SymbolicNeuralNetwork(model::Model; eqs::NamedTuple)
    SymbolicNeuralNetwork(UnknownArchitecture(), model; eqs)
end

function SymbolicNeuralNetwork(arch::Architecture; eqs::NamedTuple)
    SymbolicNeuralNetwork(arch, Chain(arch); eqs)
end

(snn::SymbolicNeuralNetwork)(x, params) = snn.functions.eval(x, params)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)

#=
function Base.show(io::IO, snn::SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
end
=#

##############

struct SymbolicModel <: Model
    model
    eval
end

(model::SymbolicModel)(x, params) = model.eval(x, params)

function NeuralNetwork(snn::SymbolicNeuralNetwork, backend::Backend, ::Type{T}; kwargs...) where T
    NeuralNetwork(snn.architecture, SymbolicModel(model(snn), functions(snn).eval), backend, T; kwargs...)
end

function NeuralNetwork(snn::SymbolicNeuralNetwork, ::Type{T}; kwargs...) where T
    NeuralNetwork(snn.architecture, SymbolicModel(model(snn), functions(snn).eval), T; kwargs...)
end

function symbolize(nn::NeuralNetwork; kwargs...)
    snn = SymbolicNeuralNetwork(nn.architecture, model(nn); kwargs...)
    NeuralNetwork(architecture(nn), SymbolicModel(model(nn), functions(snn).eval), params(nn))
end

@inline model(nn::NeuralNetwork{T, <:SymbolicModel}) where T = nn.model.model