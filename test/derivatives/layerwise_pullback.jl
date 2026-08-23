# The layerwise construction of `SymbolicPullback` has to compute *the same gradient* as the
# monolithic one it replaces — that is the whole claim, and everything else about it (that the
# expression stays small, that deeper networks build at all) is worthless without it.
#
# So the reference here is the monolithic path wherever it can build, and `Zygote` throughout. The
# cases that matter beyond agreement are the ones where the layerwise construction must *decline*:
# a model that does not decompose into layers, and a loss whose relation between prediction and
# target the pass-through stand-in misrepresents. Declining means falling back, silently and
# correctly; getting that wrong would produce a plausible but wrong gradient.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: composes_layerwise, symbolic_steps, loss_seed, loss_expression,
                              passthrough_expression, represents_loss, reference_parameters,
                              layerwise_gradient_function, monolithic_gradient_function,
                              PassThroughLayer, batched
using AbstractNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, FeedForwardLoss, NetworkLoss,
                              UnknownArchitecture, AbstractExplicitLayer, input_dimension,
                              output_dimension
using NeuralNetworkParameters: NetworkParameters, flatten, unflatten, parameterlayout
using LinearAlgebra: norm
using Symbolics
using Test
import ForwardDiff, Random, Zygote

Random.seed!(123)

# This used to be `GeometricMachineLearning.ZygotePullback`; inlined for the same reason
# `test/derivatives/pullback.jl` inlines it — so the suite does not depend on a package that depends
# on this one.
zygote_gradient(loss, ps, model, input, output) =
    params(Zygote.gradient(p -> loss(model, p, input, output), ps)[1])

maximum_difference(a, b) =
    maximum(maximum(abs, a[k][f] .- b[k][f]) for k in keys(a) for f in keys(a[k]))

gradient_of(pb, nn, input, output) = pb(params(nn), nn.model, (input, output))[2](1)

@testset "layerwise agrees with monolithic: $depth layers of width $width, $indim → $outdim" for
        (depth, width, indim, outdim) in ((2, 4, 2, 2), (3, 4, 3, 2), (4, 3, 2, 1), (2, 5, 1, 3))

    hidden = ntuple(_ -> Dense(width, width, tanh), depth - 1)
    c = Chain(Dense(indim, width, tanh), hidden[1:(end - 1)]..., Dense(width, outdim, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()

    layerwise = SymbolicPullback(snn, loss; layerwise = true)
    monolithic = SymbolicPullback(snn, loss; layerwise = false)

    for (input, output) in ((rand(indim), rand(outdim)),           # a single sample
                            (rand(indim, 1), rand(outdim, 1)),     # a batch of one
                            (rand(indim, 6), rand(outdim, 6)))     # a batch
        gl = gradient_of(layerwise, nn, input, output)
        gm = gradient_of(monolithic, nn, input, output)
        @test keys(gl) == keys(gm)
        @test maximum_difference(gl, gm) < 1e-14
        # the same *type*, not merely the same numbers: this is what a training loop consumes
        @test typeof(gl) == typeof(gm)
    end
end

@testset "layerwise agrees with Zygote on a single sample" begin
    c = Chain(Dense(3, 4, tanh), Dense(4, 4, tanh), Dense(4, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()

    input, output = rand(3, 1), rand(2, 1)
    symbolic = gradient_of(SymbolicPullback(snn, loss; layerwise = true), nn, input, output)
    @test maximum_difference(symbolic, zygote_gradient(loss, params(nn), c, input, output)) < 1e-14
end

# `SymbolicPullback` differentiates the loss of one sample and sums over the batch, so for a loss that
# is not additive over the batch the reference is that sum — the same statement
# `test/derivatives/pullback.jl` pins down for the monolithic path.
@testset "layerwise sums the per-sample gradients over a batch" begin
    c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()

    input, output = rand(3, 5), rand(2, 5)
    symbolic = gradient_of(SymbolicPullback(snn, loss; layerwise = true), nn, input, output)
    per_sample = Zygote.gradient(params(nn)) do p
        sum(loss(c, p, input[:, k:k], output[:, k:k]) for k in axes(input, 2))
    end[1]
    @test maximum_difference(symbolic, params(per_sample)) < 1e-14
end

# The network from issue #49: monolithically its gradient expression has 2·10⁸ nodes and does not
# build. `ForwardDiff` over the flat parameter vector is the reference, since there is no symbolic one.
@testset "a network the monolithic path cannot build" begin
    c = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 16, tanh), Dense(16, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()

    pb = SymbolicPullback(snn, loss; layerwise = true)
    input, output = rand(2, 4), rand(2, 4)
    symbolic = gradient_of(pb, nn, input, output)

    flat, layout = flatten(params(nn))
    reference = ForwardDiff.gradient(flat) do w
        sum(loss(c, unflatten(layout, w), input[:, k:k], output[:, k:k]) for k in axes(input, 2))
    end
    @test maximum_difference(symbolic, params(unflatten(layout, reference))) < 1e-12
end

@testset "which construction `:auto` picks" begin
    two_layers = SymbolicNeuralNetwork(Chain(Dense(2, 3, tanh), Dense(3, 2, tanh)))
    one_layer = SymbolicNeuralNetwork(Chain(Dense(2, 2, tanh)))

    @test length(symbolic_steps(two_layers)) == 2
    @test composes_layerwise(two_layers)
    # a single layer has no composition to keep out of the expression, so the monolithic path wins
    @test length(symbolic_steps(one_layer)) == 1
    @test !composes_layerwise(one_layer)

    # ... and both still produce the same gradient there
    nn = NeuralNetwork(Chain(Dense(2, 2, tanh)), Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    input, output = rand(2, 3), rand(2, 3)
    gl = gradient_of(SymbolicPullback(snn, loss; layerwise = true), nn, input, output)
    ga = gradient_of(SymbolicPullback(snn, loss), nn, input, output)
    @test maximum_difference(gl, ga) < 1e-14
end

# A model that is not a `Chain` cannot be given fresh variables at its seams, because it does not
# expose any. `:auto` has to fall back; `layerwise = true` has to say so rather than fall back
# quietly.
struct WholeChain{CT} <: AbstractNeuralNetworks.Model
    chain::CT
end

(m::WholeChain)(x, ps) = m.chain(x, ps)
AbstractNeuralNetworks.output_dimension(m::WholeChain) = output_dimension(m.chain)

@testset "a model that does not decompose" begin
    c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    reference = SymbolicNeuralNetwork(nn)
    snn = SymbolicNeuralNetwork(UnknownArchitecture(), WholeChain(c), params(reference),
                                reference.input)
    loss = FeedForwardLoss()

    @test isnothing(symbolic_steps(snn))
    @test !composes_layerwise(snn)
    @test isnothing(layerwise_gradient_function(snn, loss))
    @test_throws ArgumentError SymbolicPullback(snn, loss; layerwise = true)
end

# A loss that compares the prediction to the network's *input* reads as identically zero through a
# pass-through model, so the guessed expression is not merely unavailable but wrong. Catching that is
# the difference between falling back and returning a gradient of zero.
struct SelfLoss <: NetworkLoss end

(::SelfLoss)(model::Union{Chain, AbstractExplicitLayer},
             ps::Union{NetworkParameters, NamedTuple},
             input::AbstractArray, output::AbstractArray) = norm(model(input, ps) - input) / norm(input)

@testset "a loss the pass-through stand-in misrepresents" begin
    c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = SelfLoss()

    # the guess is identically zero — a pass-through model's prediction *is* its input — and the
    # check rejects it
    ŷ, y = Symbolics.variables(:x, 1:2), Symbolics.variables(:y, 1:2)
    guess = passthrough_expression(loss, ŷ, y)
    @test iszero(build_nn_function(guess, NetworkParameters(NamedTuple()), ŷ, y)(rand(2), rand(2),
                                                                                params(nn)))
    @test isnothing(loss_expression(loss, ŷ, y))      # nothing declared
    @test isnothing(loss_seed(loss, snn))
    @test isnothing(layerwise_gradient_function(snn, loss))
    @test_throws ArgumentError SymbolicPullback(snn, loss; layerwise = true)

    # so `:auto` falls back, and does not return the zero gradient the guess would have given
    input, output = rand(2, 1), rand(2, 1)
    fallback = gradient_of(SymbolicPullback(snn, loss), nn, input, output)
    @test maximum_difference(fallback, zygote_gradient(loss, params(nn), c, input, output)) < 1e-14
    @test maximum(abs, fallback.L1.W) > 1e-6
end

# The way out for such a loss is to declare its expression. An autoencoder loss is trained with
# `output == input`, and that is the relation the seed needs — which is exactly why a declared
# expression is used as given rather than checked against the four-argument form.
struct DeclaredSelfLoss <: NetworkLoss end

(::DeclaredSelfLoss)(model::Union{Chain, AbstractExplicitLayer},
                     ps::Union{NetworkParameters, NamedTuple},
                     input::AbstractArray, output::AbstractArray) = norm(model(input, ps) - input) / norm(input)
SymbolicNeuralNetworks.loss_expression(::DeclaredSelfLoss, ŷ, y) = norm(ŷ - y) / norm(y)

@testset "a loss that declares its expression" begin
    c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = DeclaredSelfLoss()

    @test !isnothing(loss_seed(loss, snn))
    pb = SymbolicPullback(snn, loss; layerwise = true)

    # on autoencoder data — target equal to input — the declared expression is the loss
    input = rand(2, 1)
    symbolic = gradient_of(pb, nn, input, input)
    @test maximum_difference(symbolic, zygote_gradient(loss, params(nn), c, input, input)) < 1e-14
end

# The pass-through stand-in can fail in a third way, besides being right and being wrong: a
# `NetworkLoss` is free to type its method to the model it is written for, and `AbstractNeuralNetworks`
# invites it to — the generic four-argument method is the one that says "Functor not defined". A loss
# written for a `Chain` therefore *throws* when applied to a `PassThroughLayer`, which is not a
# `Chain`. That has to count as declining, because `layerwise = :auto` promises a fallback and this
# network builds monolithically without trouble.
struct ChainOnlyLoss <: NetworkLoss end

# typed on the parameters as well, so that this is strictly more specific than the generic method
# upstream rather than ambiguous with it
(::ChainOnlyLoss)(model::Chain, ps::Union{NamedTuple, NetworkParameters},
                  input::AbstractArray, output::AbstractArray) =
    norm(model(input, ps) - output) / norm(output)

@testset "a loss the pass-through stand-in cannot even be applied to" begin
    c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = ChainOnlyLoss()

    # the guess cannot be built, which is a decline and not an error
    ŷ, y = Symbolics.variables(:x, 1:2), Symbolics.variables(:y, 1:2)
    @test_throws Exception passthrough_expression(loss, ŷ, y)
    @test isnothing(loss_seed(loss, snn))
    @test isnothing(layerwise_gradient_function(snn, loss))
    @test_throws ArgumentError SymbolicPullback(snn, loss; layerwise = true)

    # so `:auto` falls back and builds, where it used to propagate the exception, and what it falls
    # back to is the monolithic construction
    input, output = rand(2, 4), rand(2, 4)
    fallback = gradient_of(SymbolicPullback(snn, loss), nn, input, output)
    monolithic = gradient_of(SymbolicPullback(snn, loss; layerwise = false), nn, input, output)
    @test maximum_difference(fallback, monolithic) < 1e-14
    @test typeof(fallback) == typeof(monolithic)

    # this loss is not additive over a batch, so `Zygote` is the reference on a single sample only —
    # the same statement the batch testset above pins down for `FeedForwardLoss`
    one = (rand(2, 1), rand(2, 1))
    @test maximum_difference(gradient_of(SymbolicPullback(snn, loss), nn, one...),
                             zygote_gradient(loss, params(nn), c, one...)) < 1e-14
end

# `build_nn_function` takes a batch with two batch dimensions, and so does the monolithic pullback.
# The sweep cannot hand such an array to a layer — the forward pass is the layer *called* — so it lays
# the batch out flat, which the parameter gradient cannot tell apart from any other arrangement of the
# same samples.
@testset "a batch with two batch dimensions" begin
    c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()

    layerwise = SymbolicPullback(snn, loss; layerwise = true)
    monolithic = SymbolicPullback(snn, loss; layerwise = false)

    input, output = rand(3, 2, 3), rand(2, 2, 3)
    gl = layerwise.fun(input, output, params(nn))(1)
    gm = monolithic.fun(input, output, params(nn))(1)
    @test keys(gl) == keys(gm)
    @test maximum_difference(gl, gm) < 1e-14
    @test typeof(gl) == typeof(gm)

    # ... and it is the same gradient as for the same samples in one batch dimension
    flat = layerwise.fun(reshape(input, 3, 6), reshape(output, 2, 6), params(nn))(1)
    @test maximum_difference(gl, flat) < 1e-14
end

@testset "batched leaves a sample and an ordinary batch alone" begin
    sample, batch = rand(3), rand(3, 5)
    @test batched(sample) === sample
    @test batched(batch) === batch
    @test size(batched(rand(3, 2, 4))) == (3, 8)
    @test size(batched(rand(3, 2, 4, 5))) == (3, 40)
end

@testset "PassThroughLayer" begin
    layer = PassThroughLayer{3}()
    @test input_dimension(layer) == 3
    @test output_dimension(layer) == 3
    x = rand(3)
    @test layer(x, NetworkParameters(NamedTuple())) === x
end

@testset "the `layerwise` keyword is validated" begin
    snn = SymbolicNeuralNetwork(Chain(Dense(2, 2, tanh)))
    @test_throws ArgumentError SymbolicPullback(snn, FeedForwardLoss(); layerwise = :yes)
end

# The guessed loss expression is checked numerically, which needs parameters and points to check at.
# Both are deterministic: a caller who seeds the RNG and then builds a pullback must get the same data
# afterwards as one who does not build it.
@testset "building a pullback does not touch the global RNG" begin
    c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
    snn = SymbolicNeuralNetwork(NeuralNetwork(c, Float64))

    Random.seed!(42)
    without = rand(3)
    Random.seed!(42)
    SymbolicPullback(snn, FeedForwardLoss())
    @test rand(3) == without
end

# The sweep never asks the first layer for the sensitivity of the loss to its input, so that
# derivative is not generated for it — half the code of one layer, in a construction whose subject is
# build time.
@testset "the first layer has no input-sensitivity kernel" begin
    snn = SymbolicNeuralNetwork(Chain(Dense(2, 3, tanh), Dense(3, 3, tanh), Dense(3, 2, tanh)))
    steps = layerwise_gradient_function(snn, FeedForwardLoss()).steps

    @test isnothing(first(steps).dλ)
    @test all(!isnothing(step.dλ) for step in Base.tail(steps))
    @test all(!isnothing(step.dθ) for step in steps)
end

@testset "reference_parameters" begin
    snn = SymbolicNeuralNetwork(Chain(Dense(2, 3, tanh), Dense(3, 2, tanh)))
    ps = reference_parameters(snn)

    @test keys(ps) == keys(params(snn))
    @test size(ps.L1.W) == size(params(snn).L1.W)
    @test size(ps.L2.b) == size(params(snn).L2.b)
    @test eltype(ps.L1.W) == Float64
    @test ps == reference_parameters(snn)
end
