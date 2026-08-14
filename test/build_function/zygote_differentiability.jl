# `build_nn_function` evaluates a batch with an *in-place* kernel by default: it allocates the
# result and lets the generated code mutate it (see `_build_nn_function_iip`). `Zygote` does not
# support mutation, so the default result cannot be differentiated in reverse mode — which matters,
# because generated functions are used *inside* losses downstream (`GeometricMachineLearning`'s
# `HNNLoss` calls `build_nn_function(hvf, …)` and differentiates the loss with `Zygote`).
#
# `inplace = false` keeps the out-of-place path, which is differentiable at the cost of one array
# per batch column. These tests pin both halves of that contract down: that the opt-out works and
# gives the right derivative, and that the default is the mutating one (so the day someone makes it
# differentiable, this test tells them the keyword has become redundant).
#
# Forward-mode AD is unaffected: the preallocated array takes its element type from the inputs
# (`promoted_eltype`), so `ForwardDiff.Dual` numbers flow through the in-place path too.

using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, NeuralNetworkParameters,
                              FeedForwardLoss
using Test
import ForwardDiff
import Random
import Zygote

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
eq = c(snn.input, params(snn))
input = rand(3, 5)

@testset "the out-of-place path is Zygote-differentiable, reduce = $reduce" for reduce in (hcat, +)
    f = build_nn_function(eq, params(snn), snn.input; reduce = reduce, inplace = false)
    grad = Zygote.gradient(p -> sum(f(input, p)), ps)[1]
    @test grad isa Union{NamedTuple, NeuralNetworkParameters}

    # the same derivative through the chain itself, which needs no code generation at all
    reference = Zygote.gradient(p -> sum(Base.reduce(reduce, [c(input[:, k], p) for k in axes(input, 2)])), ps)[1]
    for layer in keys(ps), array in keys(ps[layer])
        @test grad[layer][array] ≈ reference[layer][array]
    end
end

@testset "the two-input out-of-place path is Zygote-differentiable" begin
    output = rand(2, 5)
    pb = SymbolicPullback(snn, FeedForwardLoss(); inplace = false)
    # differentiating the *loss* still works; what is exercised here is that the pullback itself
    # can be built and evaluated in out-of-place mode
    @test pb(ps, c, (input, output))[2](1.0) isa NamedTuple
end

# Pin the current limitation. If this starts failing the in-place path became differentiable and
# the `inplace` keyword is no longer needed for `Zygote`.
@testset "the default (in-place) path is not Zygote-differentiable" begin
    f = build_nn_function(eq, params(snn), snn.input)
    @test_throws Exception Zygote.gradient(p -> sum(f(input, p)), ps)
end

# Forward-mode AD does not care about mutation, so it must work on *both* paths: the array the
# in-place kernel writes into is allocated with the promoted element type, which is the `Dual`.
@testset "ForwardDiff works either way" begin
    W = ps.L1.W
    rewrap(w) = NeuralNetworkParameters((L1 = (W = reshape(w, size(W)...), b = ps.L1.b), L2 = ps.L2))
    grads = map((true, false)) do inplace
        f = build_nn_function(eq, params(snn), snn.input; inplace = inplace)
        ForwardDiff.gradient(w -> sum(f(input, rewrap(w))), vec(W))
    end
    reference = ForwardDiff.gradient(w -> sum(c(input, rewrap(w))), vec(W))
    @test grads[1] ≈ reference
    @test grads[2] ≈ reference
end
