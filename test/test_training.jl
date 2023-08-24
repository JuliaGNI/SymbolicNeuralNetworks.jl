using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using LinearAlgebra
using Test

@variables sx[1:2]
@variables nn


arch = HamiltonianNeuralNetwork(2)
hnn = NeuralNetwork(arch, Float32)

shnn  = symbolize(hnn, 2)

Data = ([0.1f0 0.2f0 0.3f0 0.4f0 0.5f0 0.6f0], [0.2f0 0.4f0 0.6f0 0.8f0 1.0f0 1.2f0], [1.0f0 1.0f0 1.0f0 1.0f0 1.0f0 1.0f0], [1.0f0 1.0f0 1.0f0 1.0f0 1.0f0 1.0f0])

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

using Zygote
function loss(shnn, data, batch_size = 2, params = params(shnn))
    total_loss = 0
    for i in batch_size
        q = Zygote.ignore(get_data(data, :q, i))
        p = Zygote.ignore(get_data(data, :p, i))
        total_loss += shnn([q...,p...], params)[1]
    end
    return total_loss
end

#= For Debug
loss(shnn, training_data, 2)

Loss(params, batch) = loss(shnn, training_data, batch, params)

∇Loss(params, index_batch) = Zygote.gradient(p -> Loss(p, index_batch), params)[1]

∇Loss(params(shnn), (1,2))
=#

nruns = 1000

train!(shnn, training_data, mopt, loss; ntraining = nruns, batch_size = 3, showprogress = true, timer = true)






