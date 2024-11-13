# Hamiltonian Neural Network

Here we build a Hamiltonian neural network as a symbolic neural network.

```@example hnn
using SymbolicNeuralNetworks
using GeometricMachineLearning
using AbstractNeuralNetworks: Dense, initialparameters, UnknownArchitecture, Model
using LinearAlgebra: norm
using ChainRulesCore
using KernelAbstractions
import Symbolics
import Latexify

input_dim = 2
c = Chain(Dense(input_dim, 4, tanh), Dense(4, 4, tanh), Dense(4, 1))

nn = HamiltonianSymbolicNeuralNetwork(c)
x_hvf = SymbolicNeuralNetworks.vector_field(nn)
x = x_hvf.x
hvf = x_hvf.hvf
hvf |> Latexify.latexify
```

We can now train this Hamiltonian neural network based on vector field data. As a Hamiltonian we take that of a harmonic oscillator:

```@example hnn
H(z::Array{T}) where T = sum(z.^2) / T(2)
𝕁 = PoissonTensor(input_dim)
hvf_analytic(z) = 𝕁(z)

const T = Float64
n_points = 2000
z_data = randn(T, 2, n_points)
nothing # hide
```

Next we need to define a new loss function. We do this based on the [`HamiltonianSymbolicNeuralNetwork`](@ref).

```@example hnn
struct CustomLoss{NF} <: NetworkLoss
    network_function::NF
end

hvf_function = build_nn_function(hvf, x, nn)
loss = CustomLoss(hvf_function)

function (loss::CustomLoss)(model::Chain, ps::NeuralNetworkParameters, input::AbstractVector{T}, output::AbstractVector{T}) where T
    @assert axes(input) == axes(output)
    norm(loss.network_function(input, ps) - output) / norm(output)
end

function (loss::CustomLoss)(model::Chain, ps::NeuralNetworkParameters, input::AbstractMatrix{T}, output::AbstractMatrix{T}) where T 
    @assert axes(input) == axes(output)
    sum(hcat([loss(model, ps, input[:, i], output[:, i]) for i in axes(input, 2)]...)) / sqrt(size(input, 2))
end

_reshape_to_matrix(input::AbstractArray{<:Number, 3}) = reshape(input, size(input, 1), size(input, 2) * size(input, 3))

function (loss::CustomLoss)(model::Chain, ps::NeuralNetworkParameters, input::AbstractArray{T, 3}, output::AbstractArray{T, 3}) where T
    loss(model, ps, _reshape_to_matrix(input), _reshape_to_matrix(output))
end
nothing # hide
```

We can now train the network:

```@example hnn
ps = NeuralNetworkParameters(initialparameters(c, T))
dl = DataLoader(z_data, hvf_analytic(z_data))
o = Optimizer(AdamOptimizer(T), ps)
batch = Batch(10)
const n_epochs = 100
nn_dummy = NeuralNetwork(UnknownArchitecture(), c, ps, CPU())
o(nn_dummy, dl, batch, n_epochs, loss; show_progress = false)
nothing # hide
```