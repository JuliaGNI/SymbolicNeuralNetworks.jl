using SymbolicNeuralNetworks
using GeometricMachineLearning
using Symbolics
using Test
using Distances


# Creation of the HamiltonianNeuralNetwork
arch = HamiltonianNeuralNetwork(2; nhidden = 3, width = 5)
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

function loss(shnn::NeuralNetwork{<:HamiltonianNeuralNetwork, <:SymbolicModel}, data, indexbatch, params)
    loss = 0
    vectorfield(x, p) = (shnn).model.equations.vectorfield(x,p)
    for n in indexbatch
        qₙ = get_data(data, :q, n)
        pₙ = get_data(data, :p, n)
        q̇ₙ = get_data(data, :q̇, n)
        ṗₙ = get_data(data, :ṗ, n)        
        dH = vectorfield([qₙ...,pₙ...], params)
        loss += sqeuclidean(dH[1],q̇ₙ) + sqeuclidean(dH[2],ṗₙ)
    end
    return loss
end

# Parameters of the training
opt = AdamOptimizer()
nrun = 1000

# Import of the data


# training


