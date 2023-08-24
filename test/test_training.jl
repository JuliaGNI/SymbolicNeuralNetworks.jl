using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using LinearAlgebra
using Test

@variables sx[1:2]
@variables nn


arch = HamiltonianNeuralNetwork(2)
hnn = NeuralNetwork(arch, Float64)

shnn  = symbolize(hnn)



#=
Data = ([0.1 0.2 0.3 0.4 0.5 0.6], [0.2 0.4 0.6 0.8 1.0 1.2], [1.0 1.0 1.0 1.0 1.0 1.0], [1.0 1.0 1.0 1.0 1.0 1.0])
get_Data = Dict(
    :shape => SampledData,
    :nb_points=> Data -> length(Data[1]),
    :q => (Data,n) -> Data[1][n],
    :p => (Data,n) -> Data[2][n],
    :q̇ => (Data,n) -> Data[3][n],
    :ṗ => (Data,n) -> Data[4][n],
)
training_data = TrainingData(Data, get_Data)
mopt = GradientOptimizer()

function loss(hnn, data, batch_size = 2, params = nn.params)
    total_loss = 0
    for i in batch_size
        total_loss += hnn([get_data(data, :q, i)...,get_data(data, :p, i)...], params)[1]
    end
    return total_loss
end

nruns = 0

train!(hnn2, training_data, mopt, loss; ntraining = nruns, batch_size = 3)
=#