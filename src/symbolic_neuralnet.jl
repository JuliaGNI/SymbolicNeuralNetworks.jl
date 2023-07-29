include("abstract_symbolic_neuralnet.jl")

struct SymbolicNeuralNetwork{AT, ET, EF} <: AbstractSymbolicNeuralNetwork{AT, ET}
    nn::NeuralNetwork{AT}
    est::ET
    vectorfield::EF

    function SymbolicNeuralNetwork(nn::NeuralNetwork)
        est = buildsymbolic(nn)
        eval_est = eval(est)
        eval_field = is_hamitonian(nn.architecture) ? eval(field) : missing      
        new{typeof(nn.architecture), typeof(eval_est), typeof(eval_est)}(nn, eval_est, eval_field)
    end

end

neuralnet(snn::SymbolicNeuralNetwork) = snn.nn
architecture(snn::SymbolicNeuralNetwork) = snn.nn.architecture
model(snn::SymbolicNeuralNetwork) = snn.nn.model
params(snn::SymbolicNeuralNetwork) = snn.nn.params

(snn::SymbolicNeuralNetwork)(x, params = params(shnn)) = snn.est(x, develop(params)...)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)

vectorfield(shnn::SymbolicHNN, x, params = params(shnn)) = shnn.vectorfield(x, develop(params)...)

is_hamitonian(::Architecture) = false

Symbolize(nn::NeuralNetwork) = SymbolicNeuralNetwork(nn)


function buildsymbolic(nn::NeuralNetwork)

    @variables sinput[1:dim(nn.architecture)]
    
    sparams = symbolicParams(nn)

    est = nn(sinput, sparams)

    build_function(est, sinput, develop(sparams)...)[2]

end
