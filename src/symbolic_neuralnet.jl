
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
    sparams = symbolicparameters(model)

    # générer les équations
    eval = model(sinput, sparams)

    equations = merge(NamedTuple{keys(new_eqs)}(Tuple(expand_derivatives(Symbolics.scalarize(SymbolicUtils.substitute(eq, [snn => eval]))) for eq in new_eqs)),(eval = eval,))

    # générer les codes 

    pre_code = Tuple(build_function(eq, sinput, sparams...) for eq in equations)

    code = Tuple(typeof(c) <: Tuple ? c[2] : c for c in pre_code)

    # rewrite  les codes
    rewrite_codes = NamedTuple{keys(equations)}(Tuple(rewrite_neuralnetwork(c, (sinput,), sparams) for c in code))

    # générer les funcitons
    functions = NamedTuple{keys(rewrite_codes)}(Tuple(@RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(c)) for c in rewrite_codes))

    SymbolicNeuralNetwork(arch, model, sparams, functions.eval, equations, functions)
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


struct SymbolicModel <: Model
    model
    eval
end

(model::SymbolicModel)(x, params) = model.eval(x, params)

@inline model(nn::NeuralNetwork{T, <:SymbolicModel}) where T = nn.model.model

function NeuralNetwork(snn::SymbolicNeuralNetwork, backend::Backend, ::Type{T}; kwargs...) where T
    NeuralNetwork(snn.architecture, SymbolicModel(model(snn), functions(snn).eval), backend, T; kwargs...)
end

function NeuralNetwork(snn::SymbolicNeuralNetwork, ::Type{T}; kwargs...) where T
    NeuralNetwork(snn.architecture, SymbolicModel(model(snn), functions(snn).eval), T; kwargs...)
end

function symbolize(nn::NeuralNetwork; eqs::NamedTuple)
    snn = SymbolicNeuralNetwork(nn.architecture, model(nn); eqs = eqs)
    NeuralNetwork(architecture(nn), SymbolicModel(model(nn), functions(snn).eval), params(nn))
end

function symbolize(nn::NeuralNetwork, dim::Int)
    snn = SymbolicNeuralNetwork(nn.architecture, model(nn), dim)
    NeuralNetwork(architecture(nn), SymbolicModel(model(nn), functions(snn).eval), params(nn))
end