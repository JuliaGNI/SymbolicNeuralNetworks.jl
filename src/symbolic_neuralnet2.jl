
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
    code = NamedTuple(keys(equations))(Tuple(build_function(eq, sinput, sparams...)[2] for eq in equations))

    # rewrite  les codes

    # générer les funcitons
    functions = NamedTuple{keys(code)}(Tuple(@RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(f)) for f in funcitons))


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


function SymbolicNeuralNetwork(nn::NeuralNetwork, dim::Int)
    est = buildsymbolic(nn, dim)
    new{typeof(nn.architecture), typeof(est)}(nn, est)
end

Symbolize(nn::NeuralNetwork, dim::Int) = SymbolicNeuralNetwork(nn, dim)




