

struct SymbolicNeuralNetwork{AT, ET} <: AbstractSymbolicNeuralNetwork{AT}
    nn::NeuralNetwork{AT}
    est::ET

    function SymbolicNeuralNetwork(nn::NeuralNetwork, dim::Int)
        est = buildsymbolic(nn, dim)
        eval_est = eval(est)
        new{typeof(nn.architecture), typeof(eval_est)}(nn, eval_est)
    end

end

neuralnet(snn::SymbolicNeuralNetwork) = snn.nn
AbstractNeuralNetworks.architecture(snn::SymbolicNeuralNetwork) = snn.nn.architecture
AbstractNeuralNetworks.model(snn::SymbolicNeuralNetwork) = snn.nn.model
AbstractNeuralNetworks.params(snn::SymbolicNeuralNetwork) = snn.nn.params

AbstractNeuralNetworks.dim(snn::SymbolicNeuralNetwork) = dim(snn.nn)

(snn::SymbolicNeuralNetwork)(x, params = AbstractNeuralNetworks.params(snn)) = snn.est(x, develop(params)...)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)

Symbolize(nn::NeuralNetwork, dim::Int) = SymbolicNeuralNetwork(nn, dim)

function buildsymbolic(nn::NeuralNetwork, dim::Int)

    @variables sinput[1:dim]
    
    sparams = symbolic_params(nn)

    est = nn(sinput, sparams)

    build_function(est, sinput, develop(sparams)...)[2]

end
