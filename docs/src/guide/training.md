```@meta
CurrentModule = SymbolicNeuralNetworks
```

# Training a Symbolic Neural Network

[`SymbolicPullback`](@ref) is a drop-in replacement for a `Zygote`-based pullback, so a
[`SymbolicNeuralNetwork`](@ref) can be trained with the optimizers of
[`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) without any
further ceremony.

We approximate a Gaussian on ``[-1, 1]\times[-1, 1]`` with a small feed-forward network.

```@example training
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, params, FeedForwardLoss

c = Chain(Dense(2, 3, tanh), Dense(3, 3, tanh), Dense(3, 1, tanh))
snn = SymbolicNeuralNetwork(c)
pb = SymbolicPullback(snn)
nothing # hide
```

`SymbolicPullback(snn)` uses a `FeedForwardLoss`; pass a `NetworkLoss` as the second argument for
anything else.

## How the pullback is built

The gradient a [`SymbolicPullback`](@ref) evaluates is fixed; how it is *constructed* is not, and the
difference decides which networks it can be built for at all.

The direct route is to write down one symbolic expression for the loss of the whole network and
differentiate it once per scalar parameter. Both halves of that grow badly. A `Chain`'s forward pass
is inlined layer into layer, so layer ``k``'s expression contains layer ``k-1``'s once per element it
reads and the loss expression is ``O(\mathrm{width}^\mathrm{depth})`` before anything is
differentiated; differentiating it then walks the whole of it once per parameter. Four layers of
width 16 — 626 parameters — reach a gradient expression of 2·10⁸ nodes, and the build never finishes.

Common subexpression elimination does not help with this. It runs at code generation, on an expression
that has already been built and traversed; see [Building Functions](@ref).

The way out is to stop inlining the composition. Each layer gets *fresh* symbolic variables for its
own input, so the expressions built for it refer to that layer and to nothing upstream of it, and the
composition happens when the pullback is *evaluated* — where it costs a function call per layer. Per
layer this differentiates the scalar

```math
s_k = \lambda_k \cdot f_k(x_{k-1}; \theta_k)
```

twice, in which ``\lambda_k`` stands for the sensitivity of the loss to that layer's output:
``\partial{}s_k/\partial{}x_{k-1}`` is ``\lambda_{k-1}``, and ``\partial{}s_k/\partial\theta_k``
is ``\partial{}L/\partial\theta_k``. Neither a Jacobian nor a rank-3 parameter derivative is ever
materialised, and the total symbolic material becomes a sum over layers rather than a product:

| layers | width | parameters | monolithic | layerwise |
|-------:|------:|-----------:|-----------:|----------:|
| 2 | 4 | 22 | 6 652 | 792 |
| 4 | 4 | 62 | 388 700 | 2 520 |
| 6 | 4 | 102 | 12 848 828 | 4 248 |
| 4 | 8 | 186 | 8 253 148 | 11 736 |
| 4 | 16 | 626 | 209 455 964 | 68 760 |

Counted in expression nodes by `scripts/codegen_comparison.jl`, which also times both constructions.

This is what `SymbolicPullback` does by default. The `layerwise` keyword overrides the choice:

```julia
SymbolicPullback(snn, loss; layerwise = true)    # demand it, error if it does not apply
SymbolicPullback(snn, loss; layerwise = false)   # one expression for the whole network
```

`:auto`, the default, composes layer by layer for every model that decomposes into more than one
layer — see [`composes_layerwise`](@ref) for the measured crossover and why the default does not try
to reproduce it exactly.

### Losses

The layerwise construction needs one thing the monolithic one does not: the loss as a function of the
network's *prediction*, so that the sweep has a value of ``\partial{}L/\partial\hat{y}`` to start
from. `AbstractNeuralNetworks` has no interface for that — a `NetworkLoss` is applied as
`loss(model, ps, input, output)` — so the expression is obtained by applying the loss to a
[`PassThroughLayer`](@ref), a model whose prediction is its input.

That is right for a loss which reaches its model once and compares the result to `output`, and *wrong*
for one that does something else: an autoencoder loss compares the prediction to the network's input,
and so reads through a pass-through model as identically zero. Returning a zero gradient quietly would
be the worst thing this could do, so the guessed expression is checked against the loss itself before
it is used, and the construction falls back to the monolithic one when the two disagree.

A loss can say what its expression is instead, in which case it is used as given:

```julia
SymbolicNeuralNetworks.loss_expression(::MyLoss, ŷ, y) = ...
```

See [`loss_expression`](@ref).

## The data

```@example training
using GeometricMachineLearning

x_vec = -1.0:0.1:1.0
y_vec = -1.0:0.1:1.0
xy_data = hcat([[x, y] for x in x_vec, y in y_vec]...)
f(x::Vector) = exp.(-sum(x .^ 2))
z_data = mapreduce(i -> f(xy_data[:, i]), hcat, axes(xy_data, 2))

dl = DataLoader(xy_data, z_data)
nothing # hide
```

```@example training
using CairoMakie

fig = Figure()
ax = Axis3(fig[1, 1])
surface!(x_vec, y_vec, [f([x, y]) for x in x_vec, y in y_vec]; alpha = .8, transparency = true)
fig
```

## Training

```@example training
import Random # hide
Random.seed!(123) # hide
nn_cpu = NeuralNetwork(c, CPU())
o = Optimizer(AdamOptimizer(), nn_cpu)
n_epochs = 1000
batch = Batch(10)
o(nn_cpu, dl, batch, n_epochs, pb.loss, pb; show_progress = false); # hide
@time o(nn_cpu, dl, batch, n_epochs, pb.loss, pb; show_progress = false);
nothing # hide
```

```@example training
fig = Figure()
ax = Axis3(fig[1, 1])
surface!(x_vec, y_vec, [c([x, y], params(nn_cpu))[1] for x in x_vec, y in y_vec];
         alpha = .8, colormap = :darkterrain, transparency = true)
fig
```

## Comparison with a `Zygote`-based pullback

The same training run with `GeometricMachineLearning.ZygotePullback`:

```@example training
pb2 = GeometricMachineLearning.ZygotePullback(FeedForwardLoss())
o(nn_cpu, dl, batch, n_epochs, pb2.loss, pb2; show_progress = false); # hide
@time o(nn_cpu, dl, batch, n_epochs, pb2.loss, pb2; show_progress = false);
nothing # hide
```

!!! info
    For a plain feed-forward loss like this one there is no speed-up to be had — `Zygote` handles it
    perfectly well. The case for `SymbolicNeuralNetworks` is losses that *contain a derivative of the
    network*, such as those of Hamiltonian neural networks, where reverse-mode AD has to differentiate
    through a derivative; see [Double Derivatives](@ref).
