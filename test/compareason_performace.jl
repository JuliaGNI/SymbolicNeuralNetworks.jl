using GeometricMachineLearning
using SymbolicNeuralNetworks
using Symbolics
using GeometricEquations
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: hodeensemble, hamiltonian, default_parameters


function SymbolicNeuralNetworks.symbolicparameters(d::Gradient{M, N, true}) where {M,N}
    @variables K[1:d.second_dim÷2, 1:M÷2]
    @variables b[1:d.second_dim÷2]
    @variables a[1:d.second_dim÷2]
    (weight = K, bias = b, scale = a)
end

function SymbolicNeuralNetworks.symbolicparameters(d::Gradient{M, N, false}) where {M,N}
    @variables a[1:d.second_dim÷2, 1:1]
    (scale = a, )
end


#create the object ensemble_solution
ensemble_problem = hodeensemble(tspan = (0.0,4.0))
ensemble_solution =  exact_solution(ensemble_problem ) 

include("plots.jl")


#create the data associated
training_data = TrainingData(ensemble_solution)

#creating a training sets
arch = GSympNet(2; nhidden = 4, width = 10)
sympnet = NeuralNetwork(GSympNet(2; nhidden = 4, width = 10), Float64)

ssympnet = SymbolicNeuralNetwork(arch, 2)
#ssympnet = symbolize(sympnet, 2)

nruns = 100
method = BasicSympNet()
mopt = AdamOptimizer()
training_parameters = TrainingParameters(nruns, method, mopt)

training_set = TrainingSet(ssympnet, training_parameters, training_data)

#training of the neural network
neural_net_solution = train!(training_set; showprogress = true)


H(x) = hamiltonian(x[1+length(x)÷2:end], 0.0, x[1:length(x)], default_parameters)
plot_result(training_data, neural_net_solution, H; batch_nb_trajectory = 10, filename = "GSympNet 4-10 on Harmonic Oscillator", nb_prediction = 5)