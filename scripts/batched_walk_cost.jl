# What `unflatten_batch` costs to *compile*, as a function of the width of one branch.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/batched_walk_cost.jl
#
# `unflatten_batch` is this package's own walk over a `NeuralNetworkParameters.ParameterLayout` — the
# half of `split_result` that restores the batch dimension, and the one
# `scripts/allocation_comparison.jl` calls the `[batch]` rows. That script measures what the walk
# *allocates*; this one measures what it costs to compile, which is a different quantity and moves
# for different reasons.
#
# The reason to have it: the walk across the children of one branch was a `Base.tail` recursion until
# 0.7.1, and `Base.tail` yields a new tuple type at every level — so a branch of `k` children cost `k`
# specialisations over argument types each `O(k)` long, and inference on that grows as `k³`. That is
# `NeuralNetworkParameters`' D12 (`GeometricOptimizers` catalogues it as D9), found there and fixed
# there in 0.2.2; this package had the same shape in this one walk, and for the same reason it
# matters: an equation set is as wide as the parameter set it was differentiated from.
#
# **One process per row.** Each width gets a fresh Julia, because what is being timed is a first
# call and there is only one of those per process. Run in a loop instead, the first width also pays
# for everything generic that `unflatten_batch` and `rand` and `println` need compiled, and the
# widths after it do not — which flatters every row but the first and understates the growth the
# script exists to show. The child is invoked on this same file with the width as its argument, so
# the command above stays the way to run it.
#
# The layout is built and *not* timed. Building it goes through `NeuralNetworkParameters.flatten`,
# which had its own version of this defect and no longer does, and including it here would measure
# upstream rather than this package.
#
# First call, which is compilation plus a negligible run. Julia 1.11.9, Apple M4 Max:
#
#   | children | `Base.tail` chain (0.7.0) | `@generated` (0.7.1) |
#   |---|---|---|
#   | 32 | 0.23 s | 0.12 s |
#   | 64 | 0.56 s | 0.25 s |
#   | 128 | 1.68 s | 0.39 s |
#   | 369 | 11.06 s | 1.36 s |
#
# 369 is not a synthetic worst case: it is the width of the MNIST transformer in GMLDatasets.jl, the
# set that made this a defect upstream rather than a curiosity.
#
# What the walk costs is a property of the Julia as much as of the walk, so read a run against the
# version it was run on and not against the table above. The `@generated` body on 1.13.0-rc3 is
# 0.11 s, 0.31 s, 0.82 s and 8.02 s across the same four widths — level with 1.11 at 32 children and
# six times dearer at 369. That is inference on this shape getting more expensive, not a regression
# in the walk: the chain it replaced is dearer still there, by enough that measuring it is a job for
# an idle machine and a long afternoon.
#
# Which is the other thing to know about the figures: they are wall clock, so a machine with other
# work on it inflates every row. The ratios survive that and the absolutes do not.

const WIDTHS = (32, 64, 128, 369)

# `time()` and not `@elapsed` in a loop: the point is the very first call.
function first_call(f, args...)
    t = time()
    f(args...)
    round(time() - t; digits = 2)
end

flat_set(k::Integer) =
    NetworkParameters(NamedTuple{Tuple(Symbol(:e, i) for i in 1:k)}(Tuple([float(i)] for i in 1:k)))

# The child: one width, one process, one timed call.
function measure(k::Integer)
    _, layout = flatten(flat_set(k))
    out = rand(length(layout), 3)
    println("  ", lpad(k, 4), " children: ", first_call(unflatten_batch, layout, out), " s")
end

# The driver: it loads nothing of the package itself, so that a child inherits no warmed-up state.
function drive()
    println("julia ", VERSION, " — `unflatten_batch` first call, one process per row")
    for k in WIDTHS
        run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) $k`)
    end
end

if isempty(ARGS)
    drive()
else
    using SymbolicNeuralNetworks: unflatten_batch
    using NeuralNetworkParameters: NetworkParameters, flatten
    measure(parse(Int, only(ARGS)))
end
