# What a generated function costs to call, in bytes.
#
# `type_stability.jl` pins that the built functions *infer*; nothing pinned what they *allocate*, and
# issue #55 is what that gap cost. Laying equation sets out over
# `NeuralNetworkParameters.ParameterLayout` in 0.6 — in place of the local `FlatSlice` of 0.5 — put a
# `map` over a closure on the per-call path, once per nesting level. Julia 1.11 and later elide that
# closure and 1.10 does not, so the suite was green on every version while a dependent package
# measured 1.85x the allocations on 1.10 and had to ship a version-conditional ceiling to stay green
# itself. Everything here is `Base.tail` recursion now, which does not depend on the elision.
#
# The ceilings are the same on every Julia version, deliberately. A `VERSION`-conditional ceiling is
# the accommodation issue #55 exists to remove, and one here would hide exactly the class of
# regression that issue was.
#
# Each ceiling sits between two measured figures: above the largest this call costs on any supported
# Julia — 1.10 pays 64 bytes per `reshape` where 1.11 and later pay none, which is most of the spread —
# and below what the same call cost while #55 was open. A gate that the regression it was written for
# would have passed is not a gate, so the margin is deliberately narrower than the round 1.5x it is
# tempting to reach for. Both bounds are recorded beside each number.
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
                              split_result
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
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
    @test bytes_per_call(f, batch, ps) <= 220       # measured 144 (1.12-1.13) / 128 (1.10)
end

# An `EquationSetFunction`: the same, plus `split_result` putting the flat result back into the
# nesting of the parameters. This is the path issue #55 regressed on.
@testset "an equation set" begin
    f = build_nn_function(symbolic_parameter_gradient(c(snn.input, params(snn))[1], snn), snn)

    @test bytes_per_call(f, sample, ps) <= 880      # <= 768 measured, 1056 with #55 open
    @test bytes_per_call(f, batch, ps) <= 2200      # <= 2048 measured, 2320 with #55 open

    # `split_result` on its own, on a result the kernel has already produced, so that a regression is
    # attributed to the splitting rather than to the kernel.
    @test bytes_per_call(split_result, f.layout, f.f(sample, ps)) <= 620   # <= 512, was 800
    @test bytes_per_call(split_result, f.layout, f.f(batch, ps)) <= 1300   # <= 1184, was 1424
end
