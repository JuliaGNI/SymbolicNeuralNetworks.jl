# What a generated function costs to call, in bytes.
#
# `type_stability.jl` pins that the built functions *infer*; nothing pinned what they *allocate*, and
# issue #55 is what that gap cost. Laying equation sets out over
# `NeuralNetworkParameters.ParameterLayout` in 0.6 — in place of the local `FlatSlice` of 0.5 — put a
# `map` over a closure on the per-call path, in two places: `unflatten_batch` here, and the
# `unflatten` the un-batched case delegates to upstream. Julia 1.10 does not always elide that
# closure, so the suite was green on every version while a dependent package measured 1.85x the
# allocations on 1.10 and had to ship a version-conditional ceiling to stay green itself. Both walks
# are `Base.tail` recursion now, which does not depend on the elision.
#
# The two halves are separable, and the rows below say which is which:
#
#   * the *single-sample* rows go through `NeuralNetworkParameters.unflatten` and nothing of this
#     package's, so they measure the upstream half — 1056 and 800 bytes against 768 and 512, and the
#     figure depends only on which `NeuralNetworkParameters` is resolved. `Project.toml` pins 0.2.1
#     for exactly this reason: against 0.2.0 or 0.1.1 the two ceilings here are unreachable.
#   * the *batch* rows go through `unflatten_batch`, and their figure depends only on this package —
#     2320 and 1424 against 2032 and 1136, on every `NeuralNetworkParameters` tried.
#
# The ceilings are the same on every Julia version, deliberately. A `VERSION`-conditional ceiling is
# the accommodation issue #55 exists to remove, and one here would hide exactly the class of
# regression that issue was. What separates the versions is not one thing: 1.10 costs 160 bytes more
# than 1.11 on the single-sample split — five arrays at the 32 bytes of header 1.10 adds to each —
# and 48 bytes *less* than 1.12 on the batched one. So each ceiling is set from the largest figure
# measured rather than from a rule about which version is dearest.
#
# Each ceiling sits between two measured figures: above the largest this call costs on any supported
# Julia, and below what the same call cost while #55 was open. A gate that the regression it was
# written for would have passed is not a gate, so the margin is deliberately narrower than the round
# 1.5x it is tempting to reach for. Both bounds are recorded beside each number.
#
# The figures for 1.11 are measured by hand rather than in CI, whose matrix skips it —
# `NonlinearIntegrators` records the same caveat against the budget it sets for `residual!`.
#
# A number here tripping therefore means one of two things: a genuine regression, or a new Julia that
# costs more for reasons of its own. `scripts/allocation_comparison.jl` prints the same figures broken
# out by layer, which is what tells the two apart. Re-run it across the supported versions before
# changing a ceiling, and keep the recorded bounds up to date.
#
# Every `@allocated` sits inside a function that takes what it measures as an argument. A `@testset`
# body is a closure, so an `@allocated` written directly in one measures boxed captures and reports a
# figure that has nothing to do with the call.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative, symbolic_parameter_gradient, promoted_eltype,
                              split_result, unflatten_batch
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters, flatten
using Test
import Random

Random.seed!(123)

const HIDDEN = 4
const BATCH_SIZE = 8
const REPETITIONS = 100

"Bytes per call of `f(args...)`, warmed up first so that compilation is not counted."
function bytes_per_call(f, args...)
    f(args...)
    (@allocated for _ in 1:REPETITIONS
        f(args...)
    end) ÷ REPETITIONS
end

# `promoted_eltype` has to fold to a constant: the in-place path allocates the array its kernel writes
# into *before* the kernel runs, so an element type that is not known at compile time makes that
# allocation, the kernel call and everything after it dynamically dispatched. Compared rather than
# returned, because returning a `Type` across a function barrier boxes it whether it folded or not.
eltype_folds(x, ps) = promoted_eltype(x, ps) === Float64

# `NonlinearIntegrators`' `ShallowNet` basis, the shape issue #55 was measured on.
c = Chain(Dense(1, HIDDEN, tanh), Dense(HIDDEN, 1, identity; use_bias = false))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)

sample = [0.5]
batch = rand(1, BATCH_SIZE)

@testset "the element type folds to a constant" begin
    @test bytes_per_call(eltype_folds, sample, ps) == 0
    @test bytes_per_call(eltype_folds, batch, ps) == 0
end

# A bare `InPlaceBatchedFunction`: one output array per call and nothing else.
@testset "a single equation" begin
    f = build_nn_function(derivative(Jacobian(snn)), snn)
    # not a path #55 touched; these are floor checks, so the margin is the loose one
    @test bytes_per_call(f, sample, ps) <= 220      # measured 128 (1.11-1.13) / 160 (1.10)
    @test bytes_per_call(f, batch, ps) <= 220       # measured 144 (1.11-1.13) / 128 (1.10)
end

# An `EquationSetFunction`: the same, plus `split_result` putting the flat result back into the
# nesting of the parameters. This is the path issue #55 regressed on.
@testset "an equation set" begin
    f = build_nn_function(symbolic_parameter_gradient(c(snn.input, params(snn))[1], snn), snn)

    # single sample: `NeuralNetworkParameters.unflatten`, so the upstream half of the fix
    @test bytes_per_call(f, sample, ps) <= 880      # <= 768 measured, 1056 with #55 open
    # batch: `unflatten_batch`, so this package's half
    @test bytes_per_call(f, batch, ps) <= 2200      # <= 2048 measured, 2320 with #55 open

    # `split_result` on its own, on a result the kernel has already produced, so that a regression is
    # attributed to the splitting rather than to the kernel.
    @test bytes_per_call(split_result, f.layout, f.f(sample, ps)) <= 620   # <= 512, was 800
    @test bytes_per_call(split_result, f.layout, f.f(batch, ps)) <= 1300   # <= 1184, was 1424
end

# `map` drops to `Base`'s `Any32` fallback past 32 children, which returns a tuple with no concrete
# type — so before the `Base.tail` recursion a parameter set with more than 32 layers, or a layer
# with more than 32 entries, split through a type-unstable path. Nothing about it allocated
# differently enough for a ceiling above to notice, which is why it is asserted rather than measured.
@testset "the batched walk stays inferable past 32 children" begin
    wide = NetworkParameters(NamedTuple{ntuple(i -> Symbol(:e, i), 40)}(ntuple(i -> [float(i)], 40)))
    _, layout = flatten(wide)
    out = rand(length(layout), 3)

    @test isconcretetype(only(Base.return_types(unflatten_batch, (typeof(layout), typeof(out)))))
    @test unflatten_batch(layout, out).e40 == out[40:40, :]
end
