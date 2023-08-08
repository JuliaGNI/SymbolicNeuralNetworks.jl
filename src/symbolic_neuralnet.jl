
#=
    SymbolicNeuralNetwork is a type of neural network made from another one which changes the evaluation function with a symbolic one.
=#

abstract type AbstractSymbolicNeuralNetwork{AT} <: AbstractNeuralNetwork{AT} end

struct SymbolicNeuralNetwork{AT, ET} <: AbstractSymbolicNeuralNetwork{AT}
    nn::NeuralNetwork{AT}
    est::ET

    function SymbolicNeuralNetwork(nn::NeuralNetwork, dim::Int)
        est = buildsymbolic(nn, dim)
        new{typeof(nn.architecture), typeof(est)}(nn, est)
    end

end

neuralnet(snn::SymbolicNeuralNetwork) = snn.nn
AbstractNeuralNetworks.architecture(snn::SymbolicNeuralNetwork) = snn.nn.architecture
AbstractNeuralNetworks.model(snn::SymbolicNeuralNetwork) = snn.nn.model
AbstractNeuralNetworks.params(snn::SymbolicNeuralNetwork) = snn.nn.params

AbstractNeuralNetworks.dim(snn::SymbolicNeuralNetwork) = dim(snn.nn)

(snn::SymbolicNeuralNetwork)(x, params = AbstractNeuralNetworks.params(snn)) = snn.est(x, params)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)

Symbolize(nn::NeuralNetwork, dim::Int) = SymbolicNeuralNetwork(nn, dim)

function buildsymbolic(nn::NeuralNetwork, dim::Int)

    RuntimeGeneratedFunctions.init(@__MODULE__)

    @variables sinput[1:dim]
    
    sparams = symbolic_params(nn)

    est = nn(sinput, sparams)

    code = build_function(est, sinput, develop(sparams)...)[2]

    @show rewrite_codes = rewrite_code(code, (sinput,), sparams)

    fun = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(rewrite_codes))

    fun
end
