using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using Test
using Distances
include("generation_data.jl")

# Creation of the HamiltonianNeuralNetwork
arch = HamiltonianNeuralNetwork(2; nhidden = 1, width = 2)
hnn = NeuralNetwork(arch, Float32)

# Symbolization of the HamiltonianNeuralNetwork and the vector field
@variables sx[1:2]
@variables nn(sx)[1:1]
Dx1 = Differential(sx[1])
Dx2 = Differential(sx[2])
vectorfield = [0 1; -1 0] * [Dx1(nn[1]), Dx2(nn[1])]
eqs = (x = sx, nn = nn, vectorfield = vectorfield)

shnn = symbolize(hnn; eqs = eqs)

# Definition of the loss function

function loss(shnn::NeuralNetwork{<:HamiltonianNeuralNetwork, <:SymbolicModel}, data, indexbatch = 1:get_nb_point(data), params = shnn.params)
    loss = 0
    fvectorfield(x, p) = (shnn).model.equations.vectorfield(x,p)
    for n in indexbatch
        qₙ = get_data(data, :q, n)
        pₙ = get_data(data, :p, n)
        q̇ₙ = get_data(data, :q̇, n)
        ṗₙ = get_data(data, :ṗ, n)        
        dH = fvectorfield([qₙ...,pₙ...], params)
        loss += sqeuclidean(dH[1],q̇ₙ) + sqeuclidean(dH[2],ṗₙ)
    end
    return loss
end

# Parameters of the training
opt = AdamOptimizer()
nrun = 10

# Import of the data
println("Begin generation of Data")
data = get_HNN_data(:pendulum, 10)
println("End generation of Data")


fvectorfield(x, p) = (shnn).model.equations.vectorfield(x,p)

# Training
#total_loss = train!(shnn, data, opt, loss; ntraining = nrun, batch_size = 10, showprogress = true, timer = true)

