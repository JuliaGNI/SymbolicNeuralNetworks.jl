# Replace `GeometricMachineLearning` with `GeometricOptimizers` in the docs

## Context

`docs/Project.toml` depends on `GeometricMachineLearning` (GML) and **the docs environment does not
resolve**. GML 0.6.0 — the only registered 0.6 — pins `NeuralNetworkParameters = "0.1"` and
`SymbolicNeuralNetworks = "0.6"`, while this branch is `SymbolicNeuralNetworks` 0.7.0 on
`NeuralNetworkParameters` 0.2.2. The Documentation CI job stays red until that is fixed, and it
cannot be fixed from this side: GML has a compat bound *on this package*, so a dependency in the
other direction means neither can be released without the other having been released first. The test
suite already avoids GML for exactly this reason (`test/derivatives/pullback.jl:10-14`).

### What GML is actually used for

Nothing in `src/`, `test/` or the top-level `Project.toml` depends on GML — the occurrences there
are prose comments only. In `docs/src` it appears in **one executable page**,
`docs/src/guide/training.md`, for five symbols:

| symbol | line | role |
|---|---|---|
| `DataLoader` | :147 | wraps the `(xy_data, z_data)` sample grid |
| `CPU()` | :165 | backend argument to `NeuralNetwork(c, CPU())` |
| `Optimizer` / `AdamOptimizer` | :166 | the optimizer |
| `Batch` | :168 | minibatch index sets |
| `Optimizer` functor | :169-170 | the epoch/batch training loop |
| `ZygotePullback` | :187 | the timing baseline the page compares against |

Everything else is prose: `index.md:14`, `index.md:98`, `limitations.md:97`, `training.md:9/83/184`.
`docs/make.jl` has no GML reference at all.

### Why `GeometricOptimizers` 0.6

`GeometricOptimizers` (GO) is where GML's optimizer machinery is being moved to; GML 0.5-DEV is
already a thin adapter over it (`src/optimizers/go_bridges.jl`). GO 0.6.0 on `main`
(`/Users/mkraus/Datashare/Julia/GeometricOptimizers`) has compat that matches this branch exactly —
`AbstractNeuralNetworks = "0.7"`, `NeuralNetworkParameters = "0.2.2"`, `julia = "1.11"` — and,
decisively, **it does not depend on `SymbolicNeuralNetworks`**, so the release deadlock disappears.

What it does *not* have: `DataLoader`, `Batch`, `optimization_step!`. Those stayed in GML. So
`training.md` writes its own minibatch loop.

## Decisions taken

- Keep `SymbolicPullback` as the page's subject; wrap it to meet GO's flat-gradient interface.
- Use GO's unexported `solver_step!` / `increase_iteration_number!`, with a note in the page saying
  they are internal for now.
- **No `[sources]` entry.** Write the page against the GO 0.6 API and set
  `[compat] GeometricOptimizers = "0.6"`; this lands once GO 0.6.0 is registered. Docs CI stays red
  until then — the same state as today, but with the blocker moved to something that *can* be
  released.
- Scope is `docs/Project.toml` and `docs/src/guide/training.md` only. `README.md:63-72` and
  `scripts/pullback_comparison.jl` keep their GML code for now; the prose mentions in `index.md` and
  `limitations.md` stay as they are.

## The interface to bridge

GO's `GradientFunction(F, ∇F!, ::ParameterSet)`
(`GeometricOptimizers/src/optimizers/named_tuple_wrapper.jl:19`) flattens the parameter set once and
calls `∇F!` **on the flat vector**, while `F` is wrapped as `_x -> F(unflatten(layout, _x))` and so
sees the container. `SymbolicPullback`'s pullback returns a plain `NamedTuple`
(`src/derivatives/pullback.jl:194-196`), and `ParameterSet = Union{NetworkParameters, NamedTuple}`,
so `NeuralNetworkParameters.flatten!(g, dp, layout)` copies it into `g` with no allocation.

## Changes

### 1. `docs/Project.toml`

- Replace the `GeometricMachineLearning` UUID line in `[deps]` with
  `GeometricOptimizers = "fc236c15-5557-4942-aa65-b650f329279e"`.
- Replace the `[compat]` block's GML entry and its comment with
  `GeometricOptimizers = "0.6"`, and a comment recording why: 0.6 is the first release tracking
  `AbstractNeuralNetworks` 0.7 and `NeuralNetworkParameters` 0.2, GML is gone because its compat
  bound on this package deadlocks releases in both directions, and this bound is unresolvable until
  GO 0.6.0 is registered.
- `[sources]` is unchanged (`SymbolicNeuralNetworks = {path = ".."}`).

### 2. `docs/src/guide/training.md`

**Opening prose (:7-10)** — change "the optimizers of `GeometricMachineLearning`" to
`GeometricOptimizers`, with the link updated to `JuliaGNI/GeometricOptimizers.jl`. The "without any
further ceremony" claim needs softening: there *is* ceremony now, namely flattening the gradient.

**Layers section (:83)** — leave the `SymplecticEuler` example alone. It illustrates a carrying
layer and costs nothing now that GML is not a dependency.

**The data block (:138-149)** — drop `using GeometricMachineLearning` and `DataLoader`; keep
`xy_data`/`z_data` as the plain arrays they already are.

**Training block (:162-172)** — replace with, in shape:

```julia
using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!
using NeuralNetworkParameters: flatten, flatten!, unflatten
using AbstractNeuralNetworks: NeuralNetwork, params
import Random

nn = NeuralNetwork(c)                       # no `CPU()` — that method was GML's
ps = params(nn)
_, layout = flatten(ps)

minibatch = Ref((xy_data, z_data))          # what F and ∇F! read; set per step

F(p) = pb.loss(c, p, minibatch[]...)
function ∇F!(g, v)
    dp = pb(unflatten(layout, v), c, minibatch[])[2](1.0)
    flatten!(g, dp, layout)
end

method = Adam(Float64)
o = Optimizer(ps, F; ∇F! = ∇F!, algorithm = method, linesearch = Static(1e-3))
state = OptimizerState(method, ps)

for _ in 1:n_epochs
    for idx in Iterators.partition(Random.shuffle(axes(xy_data, 2)), 10)
        minibatch[] = (xy_data[:, idx], z_data[:, idx])
        increase_iteration_number!(state)
        solver_step!(ps, state, o)
        update!(state, o, ps)
    end
end
```

Keep the existing `Random.seed!(123)` hide-line, the warm-up run and the `@time` on the second run.
The surface plot at :174-180 reads `params(nn_cpu)`; rename to `ps`.

Add a short note that `solver_step!` and `increase_iteration_number!` are not yet exported by
`GeometricOptimizers`, so the page imports them explicitly.

**Comparison section (:182-191)** — replace `GeometricMachineLearning.ZygotePullback` with the same
one-liner the test suite uses (`test/derivatives/pullback.jl:14`). `Zygote` is already a
`docs/Project.toml` dep. The second `∇F!` differs from the first only in where the gradient comes
from:

```julia
import Zygote
function ∇F_zygote!(g, v)
    p = unflatten(layout, v)
    _, back = Zygote.pullback(q -> pb.loss(c, q, minibatch[]...), p)
    flatten!(g, params(back(1.0)[1]), layout)
end
```

Run it from freshly-initialised `ps`/`state`/`Optimizer` so the two timings start from the same
point. The `!!! info` box at :193-197 is unchanged.

Factor the epoch loop into one `train!(ps, o, state)` helper used by both blocks rather than writing
it twice.

## Verification

1. `rm -f docs/Manifest.toml` first — it is gitignored and a stale local copy will not re-resolve.
   See the recorded staleness traps (a diverged `~/.julia/registries/General` clone reports
   `Unsatisfiable requirements` without saying the registry is old).
2. The environment cannot resolve until GO 0.6.0 is registered. Until then, verify against the local
   clone by temporarily adding
   `GeometricOptimizers = {path = "/Users/mkraus/Datashare/Julia/GeometricOptimizers"}` to
   `[sources]` — **and reverting it before committing.**
3. Check the two interface assumptions explicitly, since both are load-bearing:
   - `flatten!(g, dp, layout)` accepts a plain `NamedTuple` `dp` against a layout built from a
     `NetworkParameters`, and orders it identically. If it does not, wrap: `flatten!(g,
     NetworkParameters(dp), layout)`.
   - `Optimizer(ps, F; ∇F! = ...)` accepts a `NetworkParameters` (a nested `ParameterContainer`, not
     an `ArrayNamedTuple`). `GeometricOptimizers/test/network_parameters_optimizer.jl` covers the
     autodiff path for this shape; the `∇F!` path is the untested one.
4. Confirm the loss actually falls — the trained surface plot at the end of the page is the check,
   and a gradient flattened in the wrong order will still run and produce a wrong plot.
5. Build: `julia --project=docs docs/make.jl`. Watch the wall clock — GO's `solver_step!` evaluates
   the merit and runs a line search per step, which GML's `optimization_step!` did not, so 1000
   epochs × 45 batches may be materially slower. Reduce `n_epochs` if the doc build becomes
   unreasonable, and say so in the page.
6. `grep -rn GeometricMachineLearning docs/` should return only the prose hits in `index.md:14`,
   `limitations.md:97` and `training.md`'s `SymplecticEuler` paragraph.
7. Update `CHANGELOG.md`: the Open Issues → Upstream entry about waiting for a GML release is
   replaced by one about waiting for a GO 0.6.0 registration.
