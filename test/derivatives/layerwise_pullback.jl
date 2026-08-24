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
                              PassThroughLayer, batched, layer_seed, layer_step,
                              checked_layer_seed, scalar_expressions
using AbstractNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, FeedForwardLoss, NetworkLoss,
                              UnknownArchitecture, AbstractExplicitLayer, input_dimension,
                              output_dimension, layers, Initializer, NeuralNetworkBackend,
                              ArrayOrNamedTuple
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

# The error `layerwise = true` raises, caught rather than asserted with `@test_throws`, so that its
# type *and* the reason it names are both checked without building the pullback twice. Each of the
# three declines names a different one, and pinning that is what keeps a reason attached to the exit
# it comes from.
function layerwise_error(snn, loss)
    try
        SymbolicPullback(snn, loss; layerwise = true)
        nothing
    catch raised
        raised
    end
end

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
    raised = layerwise_error(snn, loss)
    @test raised isa ArgumentError
    @test occursin("does not decompose", raised.msg)
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
    raised = layerwise_error(snn, loss)
    @test raised isa ArgumentError
    @test occursin("loss cannot be expressed", raised.msg)

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
    raised = layerwise_error(snn, loss)
    @test raised isa ArgumentError
    @test occursin("loss cannot be expressed", raised.msg)

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

# Issue #54. A layer may pass data on to the next one alongside the state — `GeometricMachineLearning`'s
# `SymplecticEuler` threads the parameters of the *system* through the chain that way — in which case
# its output is a `Tuple` and not an array. The seam the layerwise construction puts between two layers
# is a plain vector of symbolic variables, so such a layer has no seed: `λ · f(x; θ)` has nothing to
# form. The chain nevertheless *decomposes*, so `composes_layerwise` says yes and `:auto` commits,
# which is how a `MethodError` used to escape a keyword that promises a fallback.
#
# These two layers are the smallest model of that chain. `Seamed` says whether they declare the seam
# interface; with `false` they do not, which is the case here — the case where it is declared is
# further down.
struct ThreadingLayer{M, N, C, Seamed} <: AbstractExplicitLayer{M, N} end
struct JoiningLayer{M, N, C, Seamed} <: AbstractExplicitLayer{M, N} end

const SeamLayer{M, N, C, S} = Union{ThreadingLayer{M, N, C, S}, JoiningLayer{M, N, C, S}}

seam_chain(indim, width, outdim, carried, seamed) =
    Chain(ThreadingLayer{indim, width, carried, seamed}(),
          JoiningLayer{width, outdim, carried, seamed}())

# The carried datum a layer supplies for itself when it is applied to a bare array, the way
# `SymplecticEuler` defaults the parameters of the system to `NullParameters`. Deterministic, for the
# reason `reference_parameters` is.
default_carried(::SeamLayer{M, N, C, S}) where {M, N, C, S} = [cospi(i / 3) for i in 1:C]

# The carried datum enters the state map, so a construction that dropped it is caught numerically and
# not merely structurally — which is the mistake the monolithic path makes on a parametrized network.
(layer::ThreadingLayer)(x::AbstractArray, ps) = layer((x, default_carried(layer)), ps)
(::ThreadingLayer)(xc::Tuple, ps) = (tanh.(ps.W * first(xc) .+ ps.b .+ sum(last(xc))), last(xc))

# Applied to a bare array the layer above defaults the carried datum, which is what lets the
# *monolithic* construction build this chain. This one only ever sees the tuple the layer before it
# produced, and so has no method for the bare vector at the seam at all — the second of the two ways a
# layer can fail to be seeded, and the one `applicable(scalar_expressions, layer(sx, ps))` would miss.
(::JoiningLayer)(xc::Tuple, ps) = tanh.(ps.W * first(xc) .+ ps.b .+ sum(last(xc)))

function AbstractNeuralNetworks.initialparameters(::Random.AbstractRNG, ::Initializer,
                                                 ::SeamLayer{M, N, C, S}, ::NeuralNetworkBackend,
                                                 ::Type{T}; kwargs...) where {M, N, C, S, T}
    (W = T[cospi((i + 2j) / 7) for i in 1:N, j in 1:M], b = T[sinpi(i / 5) for i in 1:N])
end

@testset "a layer that cannot be seeded" begin
    c = seam_chain(2, 3, 2, 2, false)
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()

    # the chain does decompose, and both layers have known dimensions — the decline is not
    # `symbolic_steps`', which is the whole point of the issue
    @test length(symbolic_steps(snn)) == 2
    @test composes_layerwise(snn)

    sparams = params(snn)
    l1, l2 = layers(c)
    @test_throws Exception layer_seed(l1, :L1, sparams.L1)      # its output is a `Tuple`
    @test_throws Exception layer_seed(l2, :L2, sparams.L2)      # its input cannot be a bare vector
    @test isnothing(checked_layer_seed(l1, :L1, sparams.L1))
    @test isnothing(checked_layer_seed(l2, :L2, sparams.L2))

    @test isnothing(layerwise_gradient_function(snn, loss))
    raised = layerwise_error(snn, loss)
    @test raised isa ArgumentError
    @test occursin("cannot be seeded", raised.msg)
    # the message names the layers, since `layerwise = true` asked for this construction by name
    @test occursin("`L1`", raised.msg) && occursin("`L2`", raised.msg)
    @test occursin("ThreadingLayer", raised.msg) && occursin("JoiningLayer", raised.msg)

    # so `:auto` builds, where it used to propagate a `MethodError`, and what it builds is the
    # monolithic construction
    input, output = rand(2, 4), rand(2, 4)
    fallback = gradient_of(SymbolicPullback(snn, loss), nn, input, output)
    monolithic = gradient_of(SymbolicPullback(snn, loss; layerwise = false), nn, input, output)
    @test maximum_difference(fallback, monolithic) < 1e-14
    @test typeof(fallback) == typeof(monolithic)

    # and it is the right gradient. `FeedForwardLoss` is not additive over a batch, so `Zygote` is the
    # reference on a single sample only
    one = (rand(2, 1), rand(2, 1))
    @test maximum_difference(gradient_of(SymbolicPullback(snn, loss), nn, one...),
                             zygote_gradient(loss, params(nn), c, one...)) < 1e-14
end

# The other half of issue #54: the seam can be *widened* to carry what a layer passes alongside the
# state, and then this chain composes layer by layer after all. Four methods say how — see
# `seam_interface` — and they are declared here for the `Seamed = true` layers only, so that the pair
# above goes on standing for a layer that has not declared them.
SymbolicNeuralNetworks.carried_variables(::SeamLayer{M, N, C, true}) where {M, N, C} =
    (Symbolics.variables(:c, 1:C),)
SymbolicNeuralNetworks.seam_value(::SeamLayer{M, N, C, true}, sx, sc) where {M, N, C} = (sx, sc)
# only the layer that returns the pair needs this one; the other returns the state alone, which is the
# default
SymbolicNeuralNetworks.state_expressions(::ThreadingLayer{M, N, C, true}, y) where {M, N, C} =
    scalar_expressions(first(y))

# `seam_arguments` has to hand the kernels a carried datum with the state's rank and batch size — the
# constraint every generated function of the package puts on its data arguments — which for data that
# is the same for the whole batch means broadcasting it out to one column per sample.
match_batch(carried, ::AbstractVector) = carried
match_batch(carried, state::AbstractMatrix) = repeat(carried, 1, size(state, 2))

SymbolicNeuralNetworks.seam_arguments(::SeamLayer{M, N, C, true}, x::Tuple) where {M, N, C} =
    (first(x), match_batch(last(x), first(x)))
SymbolicNeuralNetworks.seam_arguments(layer::SeamLayer{M, N, C, true},
                                      x::AbstractArray) where {M, N, C} =
    (x, match_batch(default_carried(layer), x))

@testset "a layer that declares the seam interface" begin
    c = seam_chain(2, 3, 2, 2, true)
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    sparams = params(snn)
    l1, l2 = layers(c)

    # both layers seed now, and the seam holds one array besides the state
    seeded = checked_layer_seed(l1, :L1, sparams.L1)
    @test !isnothing(seeded)
    sdata = seeded[3]
    @test length(sdata) == 2
    @test length(first(sdata)) == input_dimension(l1)   # the state
    @test length(last(sdata)) == 2                      # the carried datum, as the layer declared it
    @test !isnothing(checked_layer_seed(l2, :L2, sparams.L2))

    g = layerwise_gradient_function(snn, loss)
    @test !isnothing(g)
    # the first layer still gets no input-sensitivity kernel, and every step gets a parameter one
    @test isnothing(first(g.steps).dλ)
    @test all(!isnothing(step.dθ) for step in g.steps)

    layerwise = SymbolicPullback(snn, loss; layerwise = true)
    monolithic = SymbolicPullback(snn, loss; layerwise = false)

    # given a bare array every layer supplies the carried datum itself, which is the only thing the
    # monolithic construction can do — so on that input the two agree
    input, output = rand(2, 4), rand(2, 4)
    gl = gradient_of(layerwise, nn, input, output)
    gm = gradient_of(monolithic, nn, input, output)
    @test maximum_difference(gl, gm) < 1e-14
    @test typeof(gl) == typeof(gm)

    # `FeedForwardLoss` is not additive over a batch, so `Zygote` is the reference on a single sample
    one = (rand(2, 1), rand(2, 1))
    @test maximum_difference(gradient_of(layerwise, nn, one...),
                             zygote_gradient(loss, params(nn), c, one...)) < 1e-14

    # over a batch the reference is the sum of the per-sample gradients, here from `ForwardDiff` over
    # the flat parameter vector
    flat, layout = flatten(params(nn))
    reference = ForwardDiff.gradient(flat) do w
        sum(loss(c, unflatten(layout, w), input[:, k:k], output[:, k:k]) for k in axes(input, 2))
    end
    @test maximum_difference(gl, params(unflatten(layout, reference))) < 1e-12
end

# A layer that carries *nothing* says so with an empty tuple, and the seam is then the plain vector it
# always was: there is nothing to hand the generated kernels, and an empty array would not be a usable
# data argument in any case — `Symbolics.variables(:c, 1:0)` is not even a `Vector{Num}`. This is
# `SymplecticEuler` over a system with no parameters, which is what the issue's reproducer builds.
#
# The layer still has to be *given* something, though, since its functor takes the pair either way —
# so `seam_value` supplies it as a constant rather than as a variable. Nothing varies, so nothing
# needs a variable; the constant is folded into the expression like any other.
SymbolicNeuralNetworks.carried_variables(::SeamLayer{M, N, 0, true}) where {M, N} = ()
SymbolicNeuralNetworks.seam_value(layer::SeamLayer{M, N, 0, true}, sx) where {M, N} =
    (sx, default_carried(layer))
SymbolicNeuralNetworks.seam_arguments(::SeamLayer{M, N, 0, true}, x::Tuple) where {M, N} = (first(x),)
SymbolicNeuralNetworks.seam_arguments(::SeamLayer{M, N, 0, true}, x::AbstractArray) where {M, N} = (x,)

@testset "a layer that carries nothing" begin
    c = seam_chain(2, 3, 2, 0, true)
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = FeedForwardLoss()
    l1, l2 = layers(c)

    # the seam is one array again, and it is the state
    seeded = checked_layer_seed(l1, :L1, params(snn).L1)
    @test !isnothing(seeded)
    @test length(seeded[3]) == 1
    @test length(only(seeded[3])) == input_dimension(l1)
    @test !isnothing(checked_layer_seed(l2, :L2, params(snn).L2))

    # and it composes, agreeing with the monolithic path and with `Zygote`
    input, output = rand(2, 4), rand(2, 4)
    gl = gradient_of(SymbolicPullback(snn, loss; layerwise = true), nn, input, output)
    gm = gradient_of(SymbolicPullback(snn, loss; layerwise = false), nn, input, output)
    @test maximum_difference(gl, gm) < 1e-14
    @test typeof(gl) == typeof(gm)

    one = (rand(2, 1), rand(2, 1))
    @test maximum_difference(gradient_of(SymbolicPullback(snn, loss; layerwise = true), nn, one...),
                             zygote_gradient(loss, params(nn), c, one...)) < 1e-14
end

# `layer_seed` names the state `x` and the sensitivities `λ`, so a `carried_variables` that reuses one
# of those names does not declare a second array — it names the same one twice. Nothing downstream
# would notice: `Symbolics.build_function` binds it to two argument slots and the generated code reads
# both from the last one, so the kernels build, they run, and the gradient is wrong. `layer_step`
# refuses the seam, and refuses rather than declines, since this is a bug in the layer and not a chain
# to fall back on.
struct CollidingLayer{M, N} <: AbstractExplicitLayer{M, N} end

(::CollidingLayer)(xc::Tuple, ps) = tanh.(ps.W * first(xc) .+ ps.b .+ sum(last(xc)))

SymbolicNeuralNetworks.carried_variables(::CollidingLayer{M, N}) where {M, N} =
    (Symbolics.variables(:x, 1:M),)                 # `x` is what the state is called
SymbolicNeuralNetworks.seam_value(::CollidingLayer, sx, sc) = (sx, sc)
SymbolicNeuralNetworks.seam_arguments(::CollidingLayer, x::Tuple) = (first(x), last(x))

@testset "a seam whose variables are not distinct" begin
    layer = CollidingLayer{2, 3}()
    prototype = params(SymbolicNeuralNetwork(Chain(Dense(2, 3, tanh)))).L1

    # it seeds — the collision is invisible at that point, which is why the check is not there
    seeded = checked_layer_seed(layer, :L1, prototype)
    @test !isnothing(seeded)

    raised = try
        layer_step(layer, :L1, seeded)
        nothing
    catch e
        e
    end
    @test raised isa ArgumentError
    @test occursin("CollidingLayer", raised.msg)
    @test occursin("distinct", raised.msg)
end

# `AbstractNeuralNetworks`' losses take `input::ArrayOrNamedTuple`, so a model whose input carries data
# alongside the state needs a loss written for it — `GeometricMachineLearning`'s `ParametricLoss` is
# one. This is the smallest such loss, making the comparison `FeedForwardLoss` makes.
struct CarryingLoss <: NetworkLoss end

carrying_loss(model, ps, input, output) = norm(model(input, ps) - output) / norm(output)

# Two methods rather than one with `input` left untyped, for the reason `SymbolicPullback`'s functor
# has two: a method more specific than the generic one upstream in its *model* argument and less
# specific in its input would be ambiguous with it rather than an override. The second signature is
# upstream's exactly, on this loss type.
(::CarryingLoss)(model::Chain, ps::Union{NetworkParameters, NamedTuple}, input::Tuple,
                 output::AbstractArray) = carrying_loss(model, ps, input, output)
(::CarryingLoss)(model::Union{Chain, AbstractExplicitLayer}, ps::Union{NetworkParameters, NamedTuple},
                 input::ArrayOrNamedTuple, output::ArrayOrNamedTuple) =
    carrying_loss(model, ps, input, output)

# What the interface is *for*: the carried datum reaches the derivative. The monolithic construction
# traces the chain from a plain vector, so it can only ever differentiate the map in which every layer
# defaulted what it carries — whatever the caller then passes.
@testset "the carried datum reaches the derivative" begin
    c = seam_chain(2, 3, 2, 2, true)
    nn = NeuralNetwork(c, Float64)
    snn = SymbolicNeuralNetwork(nn)
    loss = CarryingLoss()

    layerwise = SymbolicPullback(snn, loss; layerwise = true)
    monolithic = SymbolicPullback(snn, loss; layerwise = false)

    carried = [0.75, -1.25]                     # deliberately not `default_carried`
    input, output = rand(2, 3), rand(2, 3)

    gl = layerwise.fun((input, carried), output, params(nn))(1)
    per_sample = Zygote.gradient(params(nn)) do p
        sum(loss(c, p, (input[:, k:k], carried), output[:, k:k]) for k in axes(input, 2))
    end[1]
    @test maximum_difference(gl, params(per_sample)) < 1e-14

    # the monolithic path gets the default-carried gradient, and it is a different one
    gm = monolithic.fun(input, output, params(nn))(1)
    @test maximum_difference(gl, gm) > 1e-6

    # and the pair goes through the functor, not only through `fun` — which is what a training loop
    # calls, and what `SymbolicPullback`'s widened input type is for
    value, back = layerwise(params(nn), c, ((input, carried), output))
    @test value ≈ loss(c, params(nn), (input, carried), output)
    @test maximum_difference(back(1), gl) < 1e-14
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

    # an input that carries data alongside the state has only its state laid out; what it carries is
    # `seam_arguments`' business
    carried = rand(2)
    state, rest = batched((rand(3, 2, 4), carried))
    @test size(state) == (3, 8)
    @test rest === carried
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
