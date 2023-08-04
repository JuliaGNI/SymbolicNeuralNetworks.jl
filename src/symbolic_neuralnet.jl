
#=
    SymbolicNeuralNetwork is a type of neural network made from another one which changes the evaluation function with a symbolic one.
=#


struct SymbolicNeuralNetwork{AT, ET, EF} <: AbstractSymbolicNeuralNetwork{AT}
    nn::NeuralNetwork{AT}
    est::ET
    fun::EF

    function SymbolicNeuralNetwork(nn::NeuralNetwork, dim::Int)
        fun, est = buildsymbolic(nn, dim)
        new{typeof(nn.architecture), typeof(est), typeof(fun)}(nn, est, fun)
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

    @variables sinput[1:dim]
    
    sparams = symbolic_params(nn)

    est = nn(sinput, sparams)

    fun = build_function(est, sinput, develop(sparams)...)[2]

    rewrite(fun, sparams), eval(rewrite(fun, sparams))

end
