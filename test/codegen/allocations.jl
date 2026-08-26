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
#     package's, so they measure the upstream half — 1056 and 800 bytes with #55 open, and the figure
#     depends only on which `NeuralNetworkParameters` is resolved. `Project.toml` pins 0.2.2 for
#     exactly this reason: against 0.2.0 or 0.1.1 the two ceilings here are unreachable. 0.2.2 moved
#     them again, and downwards — 768 to 560 and 512 to 352 — because it writes its across-children
#     walks out as `@generated` bodies instead of `Base.tail` chains and so materialises no temporary
#     tuple per branch. The ceilings below came down with them.
#   * the *batch* rows go through `unflatten_batch`, and their figure depends only on this package —
#     2320 and 1424 with #55 open, 2048 and 1184 now, on every `NeuralNetworkParameters` tried. 0.7.1
#     made that walk `@generated` too; that is a *compile*-cost fix and it moves these figures by
#     nothing, which `scripts/batched_walk_cost.jl` is the harness for.
#
# The ceilings are the same on every Julia version, deliberately. A `VERSION`-conditional ceiling is
# the accommodation issue #55 exists to remove, and one here would hide exactly the class of
# regression that issue was. Measured on 1.11.9, 1.12.6 and 1.13.0-rc2, the only spread left is 96
# bytes on the two batched rows, where 1.11 is the cheaper (1952 and 1088 against 2048 and 1184); the
# four single-sample figures are identical on all three. Each ceiling is set from the largest figure
# measured rather than from a rule about which version is dearest.
#
# Each ceiling sits between two measured figures: above the largest this call costs on any supported
# Julia, and below what the same call cost while #55 was open. A gate that the regression it was
# written for would have passed is not a gate, so the margin is deliberately narrower than the round
# 1.5x it is tempting to reach for. Both bounds are recorded beside each number.
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
    @test bytes_per_call(f, sample, ps) <= 220      # measured 128 on 1.11, 1.12 and 1.13
    @test bytes_per_call(f, batch, ps) <= 220       # measured 144 on 1.11, 1.12 and 1.13
end

# An `EquationSetFunction`: the same, plus `split_result` putting the flat result back into the
# nesting of the parameters. This is the path issue #55 regressed on.
@testset "an equation set" begin
    f = build_nn_function(symbolic_parameter_gradient(c(snn.input, params(snn))[1], snn), snn)

    # single sample: `NeuralNetworkParameters.unflatten`, so the upstream half of the fix
    @test bytes_per_call(f, sample, ps) <= 700      # <= 560 measured, 1056 with #55 open
    # batch: `unflatten_batch`, so this package's half
    @test bytes_per_call(f, batch, ps) <= 2200      # <= 2048 measured, 2320 with #55 open

    # `split_result` on its own, on a result the kernel has already produced, so that a regression is
    # attributed to the splitting rather than to the kernel.
    @test bytes_per_call(split_result, f.layout, f.f(sample, ps)) <= 450   # <= 352, was 800
    @test bytes_per_call(split_result, f.layout, f.f(batch, ps)) <= 1300   # <= 1184, was 1424
end

# `map` drops to `Base`'s `Any32` fallback past 32 children, which returns a tuple with no concrete
# type — so before this walk stopped being a `map` a parameter set with more than 32 layers, or a
# layer with more than 32 entries, split through a type-unstable path. Nothing about it allocated
# differently enough for a ceiling above to notice, which is why it is asserted rather than measured.
#
# 128 and not the 40 this used to use, because there are now two cliffs to stay clear of and they are
# in opposite directions. `Any32` is one. The other is the `Base.tail` chain that replaced the `map`:
# it costs one specialisation per child over `O(k)`-long argument types, so inference on it grows as
# `k³`, and at 128 children compiling this walk took 1.33 s against 0.40 s for the `@generated` body
# that replaced it in 0.7.1 — 8.94 s against 1.49 s at the 369 of GMLDatasets' MNIST transformer.
# `scripts/batched_walk_cost.jl` is the harness for those figures and is where a width beyond what a
# test suite should pay for belongs. What is asserted here is what a test can assert without pinning a
# wall clock: that the result is still concretely typed, and that the file completes at a width where
# the chain was already a second of compilation.
@testset "the batched walk stays inferable past 32 children" begin
    wide = NetworkParameters(NamedTuple{ntuple(i -> Symbol(:e, i), 128)}(ntuple(i -> [float(i)], 128)))
    _, layout = flatten(wide)
    out = rand(length(layout), 3)

    @test isconcretetype(only(Base.return_types(unflatten_batch, (typeof(layout), typeof(out)))))
    @test unflatten_batch(layout, out).e128 == out[128:128, :]
end
