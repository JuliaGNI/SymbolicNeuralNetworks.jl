# What `unflatten_batch` costs to *compile*, as a function of the width of one branch.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/batched_walk_cost.jl
#
# `unflatten_batch` is this package's own walk over a `NeuralNetworkParameters.ParameterLayout` — the
# half of `split_result` that restores the batch dimension, and the one `scripts/allocation_comparison.jl`
# calls the `[batch]` rows. `scripts/allocation_comparison.jl` measures what it *allocates*; this
# measures what it costs to compile, which is a different quantity and moves for different reasons.
#
# The reason to have it: the walk across the children of one branch was a `Base.tail` recursion until
# 0.7.1, and `Base.tail` yields a new tuple type at every level — so a branch of `k` children cost `k`
# specialisations over argument types each `O(k)` long, and inference on that grows as `k³`. That is
# `NeuralNetworkParameters`' D12 (`GeometricOptimizers` catalogues it as D9), found there and fixed
# there in 0.2.2; this package had the same shape in this one walk, and for the same reason it
# matters: an equation set is as wide as the parameter set it was differentiated from.
#
# The layout is built and *not* timed. Building it goes through `NeuralNetworkParameters.flatten`,
# which had its own version of this defect and no longer does, and including it here would measure
# upstream rather than this package.
#
# First call, which is compilation plus a negligible run. Julia 1.11.9, Apple M4 Max:
#
#   | children | `Base.tail` chain (0.7.0) | `@generated` (0.7.1) |
#   |---|---|---|
#   | 32 | 0.21 s | 0.16 s |
#   | 64 | 0.42 s | 0.18 s |
#   | 128 | 1.33 s | 0.40 s |
#   | 369 | 8.94 s | 1.49 s |
#
# 369 is not a synthetic worst case: it is the width of the MNIST transformer in GMLDatasets.jl, the
# set that made this a defect upstream rather than a curiosity.

using SymbolicNeuralNetworks: unflatten_batch
using NeuralNetworkParameters: NetworkParameters, flatten

# `time()` and not `@elapsed` in a loop: the point is the very first call.
function first_call(f, args...)
    t = time()
    f(args...)
    round(time() - t; digits = 2)
end

flat_set(k::Integer) =
    NetworkParameters(NamedTuple{Tuple(Symbol(:e, i) for i in 1:k)}(Tuple([float(i)] for i in 1:k)))

function main()
    println("julia ", VERSION, " — `unflatten_batch` first call")
    for k in (32, 64, 128, 369)
        _, layout = flatten(flat_set(k))
        out = rand(length(layout), 3)
        println("  ", lpad(k, 4), " children: ", first_call(unflatten_batch, layout, out), " s")
    end
end

main()
