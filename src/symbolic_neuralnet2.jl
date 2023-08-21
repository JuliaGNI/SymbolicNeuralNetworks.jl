
abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

struct SymbolicNeuralNetwork{AT,MT,PT,EVT,ET,FT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT

    eval::EVT
    equations::ET
    functions::FT
end

AbstractNeuralNetworks.architecture(snn::SymbolicNeuralNetwork) = snn.architecture
AbstractNeuralNetworks.model(snn::SymbolicNeuralNetwork) = snn.model
AbstractNeuralNetworks.params(snn::SymbolicNeuralNetwork) = snn.params

equations(snn::SymbolicNeuralNetwork) = snn.equations
functions(snn::SymbolicNeuralNetwork) = snn.functions


function SymbolicNeuralNetwork(arch::Architecture, model::Model; eqs::NamedTuple)

    @assert [:x, :nn] ⊆ keys(eqs)
    
    RuntimeGeneratedFunctions.init(@__MODULE__)

    sinput = eqs.x
    snn = eqs.nn

    new_eqs = [p for p in pairs(eqs) if p[1] ∉ [:x, :nn]]

    # générer les paramètres symboliques
    sparams = parameters(model)

    # générer les équations
    eval = model(sinput, sparams)

    equations = NamedTuple(:eval ∪ keys(new_eqs))(eval ∪ Tuple(expand_derivatives(SymbolicUtils.substitute(eq, [snn => eval]))))

    # générer les codes 
    code = Tuple(build_function(eq, sinput, sparams...)[2] for eq in equations)

    # rewrite  les codes
    rewrite_codes = NamedTuple(keys(equations))(Tuple(rewrite_neuralnetwork(c, (sinput,), sparams) for c in code))

    # générer les funcitons
    functions = NamedTuple{keys(code)}(Tuple(@RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(c)) for c in rewrite_codes))

    SymbolicNeuralNetwork(arch, model, sparams, eval, equations, functions)
end

function SymbolicNeuralNetwork(model::Model; eqs::NamedTuple)
    SymbolicNeuralNetwork(UnknownArchitecture(), model; eqs)
end

function SymbolicNeuralNetwork(arch::Architecture; eqs::NamedTuple)
    SymbolicNeuralNetwork(arch, Chain(arch); eqs)
end

(snn::SymbolicNeuralNetwork)(x, params = AbstractNeuralNetworks.params(snn)) = snn.eval(x, params)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)


function Base.show(io::IO, snn:SymbolicNeuralNetwork)
    print(io, "\nSymbolicNeuralNetwork with\n")
end


##############

function NeuralNetwork(snn::SymbolicNeuralNetwork, backend::Backend, ::Type{T}; kwargs...) where T
    NeuralNetwork(architecture(snn), functions(snn).eval, backend, T; kwargs...)
end

function NeuralNetwork(snn::SymbolicNeuralNetwork, ::Type{T}; kwargs...) where T
    NeuralNetwork(architecture(snn), functions(snn).eval, T; kwargs...)
end



function Symbolize(nn::NeuralNetwork, dim::Int)

end



