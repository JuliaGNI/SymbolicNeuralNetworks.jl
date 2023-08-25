using GeometricMachineLearning
using SymbolicNeuralNetworks
using Symbolics
using GeometricEquations
using Test
include("plots.jl")

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: hodeensemble, hamiltonian, default_parameters


# Importing Data
ensemble_problem = hodeensemble(tspan = (0.0,4.0))
ensemble_solution =  exact_solution(ensemble_problem ) 
training_data = TrainingData(ensemble_solution)

# Extension of symbolicparameters method for Gradient Layer

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

#creating of the neuralnetwork and the symbolized one
arch = GSympNet(2; nhidden = 4, width = 10, allow_fast_activation = false)
sympnet = NeuralNetwork(arch, Float64)
ssympnet = symbolize(sympnet, 2)

#parameters for the training
method = BasicSympNet()
mopt = AdamOptimizer()


function performance_symbolic(nruns::Int)
    training_parameters =TrainingParameters(nruns, method, mopt)
    training_set = TrainingSet(ssympnet, training_parameters, training_data)
    train!(training_set; showprogress = true, timer = true)
    nothing
end

function performance_withoutsymbolic(nruns::Int)
    training_parameters =TrainingParameters(nruns, method, mopt)
    training_set = TrainingSet(sympnet, training_parameters, training_data)
    train!(training_set; showprogress = true, timer = true)
    nothing
end


#Plots

function _plot(neuralnetsolution)
    H(x) = hamiltonian(x[1+length(x)÷2:end], 0.0, x[1:length(x)], default_parameters)
    plot_result(training_data, neural_net_solution, H; batch_nb_trajectory = 10, filename = "GSympNet 4-10 on Harmonic Oscillator", nb_prediction = 5)
end