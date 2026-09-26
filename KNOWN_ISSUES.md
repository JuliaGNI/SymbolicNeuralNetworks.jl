# Known issues

What is known to be wrong in `SymbolicNeuralNetworks.jl` and not fixed. An entry leaves this file
when its fix merges, and the CHANGELOG entry of the fix names its ID.

## Semantics

### K1 · `SymbolicPullback` assumes the loss is additive over the batch.

- location: `docs/src/limitations.md`
- evidence: It differentiates the loss of a
  *single* sample and sums the per-sample gradients (`reduce = +`). That equals the gradient of the
  batched loss only when the loss is a sum over samples.
  `AbstractNeuralNetworks.FeedForwardLoss` is not: it normalises by `norm(output)` taken over the
  whole batch, so for batches of more than one sample the symbolic pullback and a `Zygote` pullback
  of the same `NetworkLoss` disagree. This is pre-existing behaviour — the previous test suite only
  ever exercised batches of one — and cannot be fixed within the current design, which differentiates
  a single symbolic sample. It is documented in `docs/src/limitations.md` and pinned by a test.
- kind: defect
- found: 2026-08-14

### K2 · `FeedForwardLoss` divides by `norm(output)`, so a target that is identically zero gives `NaN`/`Inf`.

- location: —
- evidence: —
- kind: defect
- found: 2026-08-14

### K3 · The element type of an in-place result comes from the inputs, not from the expression (`promoted_eltype`).

- location: —
- evidence: A `Float32` network whose generated code contains a `Float64` literal rounds
  the literal rather than widening the result. That is the behaviour one wants for a network, but it
  is a deliberate choice rather than a derived one.
- kind: defect
- found: 2026-08-14

## Not differentiable / not supported

### K4 · The default (in-place) result cannot be differentiated by `Zygote`, because it is produced by mutation.

- location: —
- evidence: `inplace = false` is the escape hatch, at the cost of one allocation per sample. Pinned
  by a test, so the day the default becomes differentiable the keyword can be retired.
- kind: defect
- found: 2026-08-14

### K5 · A matrix-valued equation cannot be evaluated on a batch with two batch dimensions when `reduce = hcat`: concatenating the per-sample results already uses the second dimension.

- location: —
- evidence: It now
  throws a clear error; supporting it would need a different result layout.
- kind: defect
- found: 2026-08-14

### K6 · All data arguments of a generated function must have the same rank and batch size.

- location: —
- evidence: For a layer
  that carries data alongside the state (`seam_interface`) that is a constraint on what
  `seam_arguments` returns: carried data that is the same for the whole batch has to be broadcast out
  to one column per sample, and a carried datum that varies per sample cannot be combined with a state
  that has two batch dimensions.
- kind: defect
- found: 2026-08-14

### K7 · A layer that carries data alongside the state cannot be the last one in a chain, since the chain's output is what the loss and its seed compare against the target.

- location: —
- evidence: —
- kind: defect
- found: 2026-08-24

## Upstream

### K8 · [#40](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/40) — `test_symbolic_gradient2` remains disabled.

- location: —
- evidence: The blocker is `AbstractNeuralNetworks.Dense`
  computing `ps.W * x`, which has no method for a three-dimensional `x`; that is the *reference*
  implementation, not this package. A generated function handles such an input fine. Unblocking it
  needs matrix–tensor multiplication upstream (`GeometricMachineLearning` has one). The stale comment
  in the test now states this; the affected case is covered by assembling the reference sample by
  sample instead.
- kind: upstream
- found: 2026-08-14

### K9 · [#35](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/35) — partially resolved.

- location: `src/symbolic_neuralnet/symbolic_neuralnet.jl`
- evidence: `AbstractNeuralNetworks` 0.6.4 defines `input_dimension`/`output_dimension` for an `AbstractLayer`,
  but not for a `Chain`, which is what this package calls them on. The two `Chain` methods still live
  in `src/symbolic_neuralnet/symbolic_neuralnet.jl` and belong upstream.
- kind: upstream
- found: 2026-08-14

### K10 · The rewrite rules depend on undocumented properties of `Symbolics.build_function`'s output — the shape of the emitted function, that data arguments are only read one entry at a time, that `create_array` takes a type as its first argument, and that the in-place form addresses its output linearly.

- location: `test/codegen/codegen_drift.jl`
- evidence: `test/codegen/codegen_drift.jl` asserts each of them directly so that an upstream change
  fails there with a clear message, but there is no supported interface to rely on instead.
- kind: upstream
- found: 2026-08-14

### K11 · `use_base_mapreduce` is no longer triggered by anything this package generates under Symbolics 7; `Symbolics._mapreduce` came from reductions over un-scalarised `Symbolics.Arr`s, which the switch to scalar variables rules out.

- location: —
- evidence: The rule is kept as a `Zygote` safety net for
  user-supplied equations and is covered by a synthetic unit test, but not by any end-to-end one.
- kind: upstream
- found: 2026-08-14

## Housekeeping

### K12 · `scripts/pullback_comparison.jl` and the untracked `scripts/pullback_comparison_static.jl` depend on `GeometricMachineLearning`, which is in no project environment, so neither runs out of the box.

- location: `scripts/pullback_comparison.jl`
- evidence: `pullback_comparison_static.jl` additionally still imports the removed `symbolic_pullback`. See
  also [#9](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/9).
- kind: defect
- found: 2026-08-14

### K13 · The generated `api.md` is 106 KiB, above Documenter's 100 KiB warning threshold.

- location: `api.md`
- evidence: Splitting the
  `@autodocs` block per source directory would fix it.
- kind: docs
- found: 2026-08-14

### K14 · One open issue is untouched by this refactor: [#31](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/31) (convenience wrappers such as `parent(jac)` and `build_nn_function(jac, …)`).

- location: —
- evidence: —
- kind: defect
- found: 2026-08-23

### K15 · `GMLPLAN.md` step 7 names a CHANGELOG section that no longer exists.

- location: `GMLPLAN.md:174`
- evidence: The step reads "Update `CHANGELOG.md`: the Open Issues → Upstream entry about waiting
  for a GML release is replaced by one about waiting for a GO 0.6.0 registration." The Open Issues
  section is now this file, and its Upstream group holds no such entry.
- kind: found late
- found: 2026-09-26
