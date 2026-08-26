# Bytes allocated per call by a generated function, layer by layer.
#
# This is the reproducer for issue #55, which reported that on *Julia 1.10 only* the generated-kernel
# path costs 1.85x the allocations it did before 0.6 — 28 096 bytes per `NonlinearIntegrators`
# `residual!` call against 15 168, with 1.11 and later unchanged at 11 424. 1.10 is no longer a
# supported version as of 0.7.1, so that particular gap can no longer be reproduced here; the script
# is kept because the *breakdown* is what it is for, and because it is how a figure in
# `test/codegen/allocations.jl` is attributed to a half. It takes the call apart:
#
#   1. Whether `promoted_eltype` folds to a constant — the element type the in-place path has to know
#      *before* it can allocate the array its kernel writes into. It is measured through a wrapper
#      that compares the answer rather than returning it, so a folded type costs nothing and an
#      unfolded one shows up as the box it is; returning a `Type` across a function barrier allocates
#      whether or not it folded, and would report the same figure either way.
#   2. A bare `InPlaceBatchedFunction`, which shares `promoted_eltype` and nothing else.
#   3. An `EquationSetFunction`, which is that plus `split_result` -> `unflatten` over a
#      `ParameterLayout`. This is the shape `DQDθ`/`DVDθ` have downstream, and the one 0.6 changed:
#      0.5 laid equation sets out over a local `FlatSlice` instead.
#   4. `split_result` on its own, on a result the kernel has already produced.
#   5. The same equation set over a batch, which is the only shape that reaches `unflatten_batch`.
#
# (2) against (3) is what isolates the cause: if both regress it is in `promoted_eltype`, which they
# share; if only (3) does it is in the layout.
#
# The `[single]` rows against the `[batch]` ones then say *whose* layout. A single sample takes
# `split_result(layout, ::AbstractVector)`, which is `NeuralNetworkParameters.unflatten` end to end
# and moves only with the `NeuralNetworkParameters` version; a batch takes `unflatten_batch`, which
# is this package's and moves only with this package. Run the script against both to attribute a
# figure to one side or the other — the two are independent, and the fix for #55 needed both halves.
# Downstream calls `DQDθ` on a length-one `Vector`, so the residual it reports is the single-sample
# row and the upstream half is what moves it — which is why `NeuralNetworkParameters` 0.2.2, whose
# written-out walks took that row from 768 bytes to 560, moves the downstream figure and this
# package's own release does not.
#
# The network is the one the downstream measurement uses — `NonlinearIntegrators`' `ShallowNet` basis,
# `Chain(Dense(1, S, σ), Dense(S, 1, identity; use_bias = false))` at `S = 4`, called on a single
# sample held in a length-one `Vector`.
#
# Every `@allocated` sits inside a function that takes what it measures as an argument. A `@allocated`
# written at the top level, or in the body of a `let` or a `@testset`, measures a closure over boxed
# captures instead and reports a figure that has nothing to do with the call.
#
# Run with
#     julia --project=. scripts/allocation_comparison.jl
# on each Julia version of interest; the point of the script is the difference between them.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative, symbolic_parameter_gradient, promoted_eltype,
                              split_result
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters
using Printf
import Random

Random.seed!(123)

const HIDDEN = 4          # `S` downstream
const BATCH_SIZE = 8      # `R` downstream: the quadrature nodes of one residual
const REPETITIONS = 100   # `@allocated` counts whole bytes, so a single call rounds badly

# `NonlinearIntegrators` uses `x -> max(zero(x), x)^3`; `tanh` generates comparable code and keeps the
# script's output comparable with `codegen_comparison.jl`.
network() = Chain(Dense(1, HIDDEN, tanh), Dense(HIDDEN, 1, identity; use_bias = false))

# Compares rather than returns, so that a `promoted_eltype` that folded to a constant leaves nothing
# behind. See the header.
eltype_folds(x, ps) = promoted_eltype(x, ps) === Float64

"Bytes per call of `f(args...)`, warmed up first so that compilation is not counted."
function bytes_per_call(f, args...; repetitions::Int = REPETITIONS)
    f(args...)
    (@allocated for _ in 1:repetitions
        f(args...)
    end) ÷ repetitions
end

function measurements()
    c = network()
    snn = SymbolicNeuralNetwork(c)
    nn = NeuralNetwork(c)
    ps = params(nn)

    sinput, sparams = snn.input, params(snn)

    # `DQDθ` downstream: the gradient of a *scalar* entry of the output with respect to the
    # parameters, so `symbolic_parameter_gradient` returns one parameter-shaped set rather than an
    # array of them, and `build_nn_function` takes its `EquationSet` method.
    dqdθ = build_nn_function(symbolic_parameter_gradient(c(sinput, sparams)[1], snn), snn)
    # `V_func` downstream: a 1x1 array of `Num`, so this is a bare `InPlaceBatchedFunction`.
    v_func = build_nn_function(derivative(Jacobian(snn)), snn)

    sample = [0.5]
    batch = rand(1, BATCH_SIZE)

    # a result of the shape the kernel produces, so `split_result` can be measured on its own
    flat_sample = dqdθ.f(sample, ps)
    flat_batch = dqdθ.f(batch, ps)

    (("promoted_eltype folds",        bytes_per_call(eltype_folds, sample, ps)),
     ("V_func(x, ps)      [single]",  bytes_per_call(v_func, sample, ps)),
     ("DQDθ(x, ps)        [single]",  bytes_per_call(dqdθ, sample, ps)),
     ("split_result       [single]",  bytes_per_call(split_result, dqdθ.layout, flat_sample)),
     ("V_func(x, ps)      [batch]",   bytes_per_call(v_func, batch, ps)),
     ("DQDθ(x, ps)        [batch]",   bytes_per_call(dqdθ, batch, ps)),
     ("split_result       [batch]",   bytes_per_call(split_result, dqdθ.layout, flat_batch)))
end

println("Julia $(VERSION), hidden width $(HIDDEN), batch $(BATCH_SIZE), " *
        "$(REPETITIONS) repetitions per figure\n")
@printf("%-30s %12s\n", "call", "bytes/call")
for (name, bytes) in measurements()
    @printf("%-30s %12d\n", name, bytes)
end
