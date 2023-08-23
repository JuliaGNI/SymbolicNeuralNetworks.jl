
abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

struct SymbolicNeuralNetwork{AT,MT,PT,EVT,ET,FT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT

    eval::EVT
    equations::ET
    functions::FT
end

AbstractNeuralNetworks.architecture(snn::SymbolicNeuralNetwork) = snn.nn.architecture
AbstractNeuralNetworks.model(snn::SymbolicNeuralNetwork) = snn.nn.model
AbstractNeuralNetworks.params(snn::SymbolicNeuralNetwork) = snn.nn.params


function SymbolicNeuralNetwork(arch::Architecture, model::Model, dim::Int; eq_loss::NamedTuple = NamedTuple()) where {T}

    RuntimeGeneratedFunctions.init(@__MODULE__)

    @variables sinput[1:dim]

    #sparams = ?     # initialize params params = initialparameters(backend, T, model; kwargs...)

    equs = (eval = nn(sinput, sparams), )

    code = (eval = build_function(est, sinput, sparams...)[2], )

    rewrite_codes = rewrite_neuralnetwork(code, (sinput,), sparams)

    fun = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(rewrite_codes))

    fun

    # create neural network
    SymbolicNeuralNetwork(arch, model, params, eval, equations, functions)
end

function SymbolicNeuralNetwork(arch::Architecture, backend::Backend; kwargs...) where {T}
    SymbolicNeuralNetwork(arch, Chain(arch), backend; kwargs...)
end

function SymbolicNeuralNetwork(model::Model, backend::Backend; kwargs...) where {T}
    SymbolicNeuralNetwork(UnknownArchitecture(), model, backend; kwargs...)
end

function SymbolicNeuralNetwork(nn::Union{Architecture, Chain, GridCell}; kwargs...) where {T}
    SymbolicNeuralNetwork(nn, CPU(); kwargs...)
end

function SymbolicNeuralNetwork(nn::NeuralNetwork, dim::Int)
    est = buildsymbolic(nn, dim)
    new{typeof(nn.architecture), typeof(est)}(nn, est)
end

Symbolize(nn::NeuralNetwork, dim::Int) = SymbolicNeuralNetwork(nn, dim)



(snn::SymbolicNeuralNetwork)(x, params = AbstractNeuralNetworks.params(snn)) = snn.est(x, params)

apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)


function buildsymbolic(nn::NeuralNetwork, dim::Int)

    RuntimeGeneratedFunctions.init(@__MODULE__)

    @variables sinput[1:dim]
    
    sparams = symbolic_params(nn)

    est = nn(sinput, sparams)

    code = build_function(est, sinput, develop(sparams)...)[2]

    rewrite_codes = rewrite_neuralnetwork(code, (sinput,), sparams)

    fun = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(rewrite_codes))

    fun
end
