@doc raw"""
    SymbolicPullback <: AbstractPullback

The *symbolic pullback* of a loss function: it evaluates the loss and the derivative of the loss
with respect to the network parameters from generated code instead of by automatic differentiation.

# Constructors

    SymbolicPullback(nn, loss)
    SymbolicPullback(nn)

Build the pullback of `loss` (an `AbstractNeuralNetworks.NetworkLoss`, by default a
`FeedForwardLoss`) for the [`SymbolicNeuralNetwork`](@ref) `nn`.

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using AbstractNeuralNetworks: params
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
pb = SymbolicPullback(snn, FeedForwardLoss())
ps = params(nn)
typeof(pb(ps, nn.model, (rand(2), rand(1)))[2](1))

# output

@NamedTuple{L1::@NamedTuple{W::Matrix{Float64}, b::Vector{Float64}}}
```

# Keyword Arguments

- `cse`: perform *common subexpression elimination* when generating code (default `true`). This
  matters a lot here: without it every one of the `2 * n_layers` generated blocks re-emits the
  entire forward pass, which makes the code for networks with more than one hidden layer
  intractably large. See [`build_kernel`](@ref).
- `inplace`: evaluate a batch with an in-place kernel (default `true`). The pullback is the end of
  the differentiation chain, so nothing differentiates through it and the default is what you want
  here; `inplace = false` exists for symmetry with [`build_nn_function`](@ref).

# Implementation

An instance stores

- `loss`: the `NetworkLoss`,
- `fun`: a [`ParameterGradient`](@ref) that produces the pullback function.

Calling the functor on `ps`, `model` and an `(input, output)` tuple returns

```julia
pullback.loss(model, ps, input, output), pullback.fun(input, output, ps)
```

where the second entry is again a function — of the *output sensitivities*.

# Extended help

!!! info "Reverse Accumulation"
    In machine learning we typically do [reverse accumulation](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_and_reverse_accumulation) to perform automatic differentiation (AD).
    Assuming we are given a function that is the composition of simpler functions ``f = f_1\circ{}f_2\circ\cdots\circ{}f_n:\mathbb{R}^n\to\mathbb{R}^m`` *reverse differentiation* starts with *output sensitivities* and then successively feeds them through ``f_n``, ``f_{n-1}`` etc. So it does:
    ```math
    (\nabla_xf)^T = (\nabla_{x}f_1)^T(\nabla_{f_1(x)}f_2)^T\cdots(\nabla_{f_{n-1}(\cdots{}x)}f_n)^T(do),
    ```
    where ``do\in\mathbb{R}^m`` are the *output sensitivities* and the jacobians are stepwise multiplied from the left. So we propagate from the output stepwise back to the input. If we have ``m=1``, i.e. if the output is one-dimensional, then the *output sensitivities* may simply be taken to be ``do = 1``.

A `NetworkLoss` is scalar-valued, so the extra step of returning a function of the output
sensitivities is not strictly necessary here — the equivalent of `pb.fun(input, output, ps)(1)`
could be stored directly. It is however customary for a pullback to return a callable, which is why
this package does so too.

The pullback is the derivative of a loss *summed over the batch*, which is why the generated
function is built with `reduce = +`:

```jldoctest
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolic_parameter_gradient
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, FeedForwardLoss, params, output_dimension
using Symbolics
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = NeuralNetwork(c)
snn = SymbolicNeuralNetwork(nn)
loss = FeedForwardLoss()
input_output = (rand(2), rand(1))

pb_values = SymbolicPullback(snn, loss)(params(nn), nn.model, input_output)[2](1)

soutput = Symbolics.variables(:y, 1:output_dimension(nn.model))
gradient = symbolic_parameter_gradient(loss(nn.model, params(snn), snn.input, soutput), snn)
pb_values2 = build_nn_function(gradient, params(snn), snn.input, soutput; reduce = +)(input_output..., params(nn))

pb_values == params(pb_values2)

# output

true
```
"""
struct SymbolicPullback{NNLT, FT} <: AbstractPullback{NNLT}
    loss::NNLT
    fun::FT
end

function SymbolicPullback(nn::SymbolicNeuralNetwork, loss::NetworkLoss; cse::Bool = true, inplace::Bool = true)
    soutput = Symbolics.variables(:y, 1:output_dimension(nn.model))
    symbolic_loss = loss(nn.model, params(nn), nn.input, soutput)
    gradient = symbolic_parameter_gradient(symbolic_loss, nn)
    # `reduce = +`: the loss of a batch is the sum of the losses of its samples, so its gradient is
    # the sum of the per-sample gradients.
    gradient_function = build_nn_function(gradient, params(nn), nn.input, soutput;
                                          reduce = +, cse = cse, inplace = inplace)
    SymbolicPullback(loss, ParameterGradient(gradient_function))
end

SymbolicPullback(nn::SymbolicNeuralNetwork; kwargs...) =
    SymbolicPullback(nn, AbstractNeuralNetworks.FeedForwardLoss(); kwargs...)

"""
    ParameterGradient(gradient_function)

What a [`SymbolicPullback`](@ref) stores in its `fun` field. Applying it to an input, an output and
the network parameters returns the [`PullbackFunction`](@ref) for those.
"""
struct ParameterGradient{FT} <: Function
    gradient_function::FT
end

(g::ParameterGradient)(input, output, parameters) = PullbackFunction(g.gradient_function, input, output, parameters)

"""
    PullbackFunction(gradient_function, input, output, parameters)

The function a [`SymbolicPullback`](@ref) returns as the second entry of its result. It takes the
*output sensitivities* — which it ignores, as the loss is scalar-valued, see the extended help of
[`SymbolicPullback`](@ref) — and returns the derivative of the loss with respect to the network
parameters, as a `NamedTuple`.
"""
struct PullbackFunction{FT, IT, OT, PT} <: Function
    gradient_function::FT
    input::IT
    output::OT
    parameters::PT
end

function (pb::PullbackFunction)(::Union{Real, AbstractArray{<:Real}})
    params(pb.gradient_function(pb.input, pb.output, pb.parameters))
end

function (pullback::SymbolicPullback)(ps, model, input_output::Tuple{<:ArrayOrNamedTuple, <:ArrayOrNamedTuple})::Tuple
    pullback.loss(model, ps, input_output...), pullback.fun(input_output..., ps)
end
