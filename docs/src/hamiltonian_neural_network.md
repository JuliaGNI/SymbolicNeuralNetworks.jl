# Hamiltonian Neural Network

Here we build a Hamiltonian neural network as a symbolic neural network.

```@example hnn
using SymbolicNeuralNetworks
using GeometricMachineLearning
using AbstractNeuralNetworks: Dense, initialparameters, UnknownArchitecture, AbstractExplicitLayer
using LinearAlgebra: norm
import Symbolics

input_dim = 2
d = Dense(input_dim, 1, tanh)

nn = HamiltonianSymbolicNeuralNetwork(d)

nn.equations.hvf
```

We can now train this Hamiltonian neural network based on vector field data. As a Hamiltonian we take that of a harmonic oscillator:

```@example hnn
H(z::Array{T}) where T = sum(z.^2) / T(2)
𝕁 = PoissonTensor(input_dim)
hvf(z) = 𝕁(z)

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

function (loss::CustomLoss)(model::AbstractExplicitLayer, ps::NamedTuple, input::Array{T}, output::Array{T}) where T
    norm(loss.network_function[1](input, ps) - output)
end

loss = CustomLoss(nn.functions.hvf)
nothing # hide
```

We can now train the network:

```@example hnn
ps = initialparameters(d, T)
dl = DataLoader(z_data, hvf(z_data))
o = Optimizer(AdamOptimizer(T), ps)
batch = Batch(10)
const n_epochs = 10
nn_dummy = NeuralNetwork(UnknownArchitecture(), d, ps, CPU())
o(nn_dummy, dl, batch, n_epochs, loss; show_progress = false)
nothing # hide
```