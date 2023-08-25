struct SymbolicModel <: Model
    model
    eval
    equations
end

(model::SymbolicModel)(x, params) = model.eval(x, params)
@inline equations(model::SymbolicModel) = model.equations
@inline equations(model::SymbolicModel, key::Symbol, x, params) = equations(model)[key](x, params)

@inline model(nn::NeuralNetwork{T, <:SymbolicModel}) where T = nn.model.model

function NeuralNetwork(snn::SymbolicNeuralNetwork, backend::Backend, ::Type{T}; kwargs...) where T
    NeuralNetwork(snn.architecture, SymbolicModel(model(snn), functions(snn).eval, functions(snn)), backend, T; kwargs...)
end

function NeuralNetwork(snn::SymbolicNeuralNetwork, ::Type{T}; kwargs...) where T
    NeuralNetwork(snn.architecture, SymbolicModel(model(snn), functions(snn).eval, functions(snn)), T; kwargs...)
end

function symbolize(nn::NeuralNetwork; eqs::NamedTuple)
    snn = SymbolicNeuralNetwork(nn.architecture, model(nn); eqs = eqs)
    NeuralNetwork(architecture(nn), SymbolicModel(model(nn), functions(snn).eval, functions(snn)), params(nn))
end

function symbolize(nn::NeuralNetwork, dim::Int)
    snn = SymbolicNeuralNetwork(nn.architecture, model(nn), dim)
    NeuralNetwork(architecture(nn), SymbolicModel(model(nn), functions(snn).eval, functions(snn)), params(nn))
end