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

input_dim = 2
d = Chain(Dense(input_dim, 10, tanh), Dense(10, 1))

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

parallelized_expression, pb = parallelize_expression(nn.functions.hvf[1])

### parallelize pullback proof of concept
function parallelize_pullback!(parallelized_expression, pb)
    @eval function ChainRulesCore.rrule(::typeof(parallelized_expression), input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        output = parallelized_expression(input, ps)
        function parallelized_expression_pullback(doutput::AT)
            f̄ = NoTangent()
            backend = KernelAbstractions.get_backend(doutput)
            dinput = zero(input)
            dnt = [deepcopy(ps) for _ ∈ axes(input, 2), _ ∈ axes(input, 3)]
            kernel! = SymbolicNeuralNetworks.parallelize_expression_differential_kernel!(backend)
            kernel!(dinput, dnt, doutput, input, ps, pb; ndrange = (size(input, 2), size(input, 3)))
            dnt_final = SymbolicNeuralNetworks._sum(dnt)
            f̄, dinput, dnt_final
        end
        output, parallelized_expression_pullback
    end
    nothing
end
parallelize_pullback!(parallelized_expression, pb)
###

loss = CustomLoss(parallelized_expression)

function (loss::CustomLoss)(model::Model, ps::NamedTuple, input::Array{T}, output::Array{T}) where T
    norm(loss.network_function(input, ps) - output)
end
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