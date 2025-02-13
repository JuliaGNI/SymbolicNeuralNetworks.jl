# Hamiltonian Neural Network

Here we build a Hamiltonian neural network as a symbolic neural network.

```julia hnn
using SymbolicNeuralNetworks
using GeometricMachineLearning
using AbstractNeuralNetworks: Dense, UnknownArchitecture, Model
using LinearAlgebra: norm
using ChainRulesCore
using KernelAbstractions
import Symbolics
import Latexify

input_dim = 2
c = Chain(Dense(input_dim, 4, tanh), Dense(4, 1, identity; use_bias = false))

nn = HamiltonianSymbolicNeuralNetwork(c)
x_hvf = SymbolicNeuralNetworks.vector_field(nn)
x = x_hvf.x
hvf = x_hvf.hvf
hvf |> Latexify.latexify
```

We can now train this Hamiltonian neural network based on vector field data. As a Hamiltonian we take that of a harmonic oscillator:

```julia hnn
H(z::Array{T}) where T = sum(z.^2) / T(2)
𝕁 = PoissonTensor(input_dim)
hvf_analytic(z) = 𝕁(z)

const T = Float64
n_points = 2000
z_data = randn(T, 2, n_points)
nothing # hide
```

We now specify a pullback `HamiltonianSymbolicNeuralNetwork`

```julia hnn
_pullback = SymbolicPullback(nn)
nothing # hide
```

We can now train the network:

```julia hnn
ps = NeuralNetwork(c, T).params
dl = DataLoader(z_data, hvf_analytic(z_data))
o = Optimizer(AdamOptimizer(.01), ps)
batch = Batch(200)
const n_epochs = 150
nn_dummy = NeuralNetwork(UnknownArchitecture(), c, ps, CPU())
o(nn_dummy, dl, batch, n_epochs, _pullback.loss, _pullback; show_progress = true)
nothing # hide
```

We now integrate the vector field:

```julia hnn
using GeometricIntegrators
hvf_closure(input) = build_nn_function(hvf, x, nn)(input, nn_dummy.params)
function v(v, t, q, params)
    v .= hvf_closure(q)
end
pr = ODEProblem(v, (0., 500.), 0.1, [1., 0.])
sol = integrate(pr, ImplicitMidpoint())
```

```julia hnn
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, [sol.q[i][1] for i in axes(sol.t, 1)].parent, [sol.q[i][2] for i in axes(sol.t, 1)].parent)
```

We also train a non-Hamiltonian vector field on the same data for comparison:

```julia hnn
c_nh = Chain(Dense(2, 10, tanh), Dense(10, 4, tanh), Dense(4, 2, identity; use_bias = false))
nn_nh = NeuralNetwork(c_nh, CPU())
o = Optimizer(AdamOptimizer(T), nn_nh)
o(nn_nh, dl, batch, n_epochs * 10, FeedForwardLoss()) # we train for times as long as before
```

We now integrate the vector field and plot the solution:

```julia hnn
vf_closure(input) = c_nh(input, nn_nh.params)
function v_nh(v, t, q, params)
    v .= vf_closure(q)
end
pr = ODEProblem(v_nh, (0., 500.), 0.1, [1., 0.])
sol_nh = integrate(pr, ImplicitMidpoint())

lines!(ax, [sol_nh.q[i][1] for i in axes(sol_nh.t, 1)].parent, [sol_nh.q[i][2] for i in axes(sol_nh.t, 1)].parent)
```