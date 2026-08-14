# Changelog

All notable changes to `SymbolicNeuralNetworks.jl` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — unreleased

A refactor of the whole package for robustness, correctness and clarity. The exported surface —
`SymbolicNeuralNetwork`, `AbstractSymbolicNeuralNetwork`, `build_nn_function`, `SymbolicPullback` —
is unchanged in name, but almost everything below it moved. There are no deprecation shims.

Resolves [#14](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/14),
[#29](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/29),
[#39](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/39),
[#43](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/43) and
[#44](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/44).

### Breaking

- **Symbolic variables are now arrays of scalar variables**
  ([#14](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/14)). `SymbolicNeuralNetwork`
  builds them with `Symbolics.variables` instead of `@variables x[1:n]`, so `nn.input` is a
  `Vector{Num}` rather than a `Symbolics.Arr` and each parameter leaf is an `Array{Num}`. Printed
  names change accordingly (`x₁` instead of `sinput[1]`, `W_1₁ˏ₁` instead of `W_1[1, 1]`).

  `Symbolics` cannot differentiate with respect to an entry of a `Symbolics.Arr` without scalarising
  it, and `Symbolics.build_function` cannot generate code for an expression containing one; using
  scalar variables removes both problems, along with every `Symbolics.scalarize`/`collect` call the
  package used to need. Reductions over the network output — `sum(c(nn.input, params(nn)))` — now
  work; they previously produced code that could not run.

  A `Symbolics.Arr` handed to `build_nn_function` is still accepted and scalarised on the way in.

- **`build_nn_function` returns a named `struct`, not a closure**
  ([#43](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/43)). The result is an
  `InPlaceBatchedFunction` or an `OutOfPlaceBatchedFunction` (both `<: AbstractBatchedFunction`), so
  its type is concrete, inferable and can be stored in a typed field.

- **`symbolic_pullback` is renamed to `symbolic_parameter_gradient`**, and its return value changed:
  for a *scalar* expression it now returns the parameter-shaped gradient directly, instead of a
  one-element array containing it. Only an array-valued expression gives an array of gradients.

- **`input_dimension`/`output_dimension` come from `AbstractNeuralNetworks`** (≥ 0.6.4) rather than
  being defined here; this package only adds the `Chain` methods
  ([#35](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/35)).

- **`SymbolicNeuralNetwork(::AbstractExplicitLayer)` wraps the layer in a `Chain`**, so its
  parameters are nested under `:L1` like those of any other model. The constructor previously threw
  a `MethodError`.

- **`Jacobian(f, nn)` flattens a non-vector `f` with `vec`**, so the rows of the result are indexed
  by `vec(f)`. For the documented case of a vector-valued `f` nothing changes.

- **Compat bound raised to `AbstractNeuralNetworks = "0.6.4"`** (was `"0.3, 0.4, 0.5, 0.6"`). 0.3
  and 0.4 are missing `GenericActivation`/`TanhActivation`, so the package never loaded against
  them; 0.6.4 is the first version providing `input_dimension`/`output_dimension`.

- **The generated kernels reserve the argument names `out`, `x1`, `x2`, `ps` and `k`.** A symbolic
  variable that is *passed* to `build_nn_function` may be named anything, as
  `Symbolics.build_function` turns it into an argument and renames it; a variable left **free** in
  the equation carrying one of those names is rejected with an error when the function is built.

- **Renamed**

  | before | after |
  |--------|-------|
  | `symbolic_pullback` | `symbolic_parameter_gradient` |
  | `symbolize!` | `symbolic_variables` / `symbolic_variables!` |
  | `symboliccounter!` | `next_name!` |
  | `flatten_eqs` | `flatten_equations` |
  | `EqT` | `SymbolicExpression` |
  | `_build_nn_function` | `build_kernel` |
  | `_build_nn_function_iip` | `build_kernel!` |
  | `build_function_generated` | `generated_expression` |
  | `fix_create_array` | `use_generic_array_constructor` |
  | `fix_map_reduce` | `use_base_mapreduce` |
  | `make_kernel`, `make_kernel2` | `index_by_batch` |
  | `make_kernel_iip`, `make_kernel_iip2`, `redirect_output_writes` | `accumulate_into_output` |
  | `rewrite_arguments`, `rewrite_arguments2`, `modify_input_arguments`, `modify_input_arguments_iip`, `modify_input_arguments2`, `modify_input_arguments_iip2` | `argument_substitutions` + `substitute_symbols` |

- **Removed** (no replacement needed): `_build_nn_function_per_leaf`, `function_valued_parameters`,
  `apply_element_wise` and its `@generated` methods, `strip_of_val`, `generate_symbols`,
  `_get_params`, `_get_contents`, `_reduce`, `_reduce_iip`, `_modify_integer`, `_modify_integer2`,
  `_oop_batch_wrapper`, `_iip_batch_wrapper`, `_oop_batch_wrapper2`, `_iip_batch_wrapper2`,
  `optional_reshape`, and `apply(::AbstractSymbolicNeuralNetwork, …)`.

- **Files moved.** `src/build_function/` → `src/codegen/`; `src/utils/loss_adjustment.jl` →
  `src/losses.jl`; `src/custom_definitions_and_extensions/` → `src/symbolic_expressions.jl`;
  `src/symbolic_neuralnet/symbolize.jl` → `src/symbolic_neuralnet/symbolic_variables.jl`. `test/`
  mirrors `src/`.

### Fixed

- A three-dimensional input with `reduce = +` threw a `DimensionMismatch` in the one-data-argument
  case; the two-argument case handled it. The rank handling is now written once for both.
- A symbolic input array not literally named `sinput` threw
  `AssertionError: The first input arguments must be ˍ₋out and sinput`, because the generated code
  was matched by name in its printed form. Arguments are matched by position now.
- The two-data-argument methods required both arguments to have the *same* type (`where AT`), so a
  `Vector` input with a `SubArray` target failed. They now only require the same rank.
- `apply(snn, x, …)` dispatched to a functor that does not exist and threw a `MethodError`.
- `SymbolicNeuralNetwork(::AbstractExplicitLayer)` threw a `MethodError`.
- `sum(c(nn.input, params(nn)))` silently generated code that raised an `UndefVarError` when called.
- The `Project.toml` claimed compatibility with `AbstractNeuralNetworks` 0.3 and 0.4, against which
  the package does not load at all.
- The double-derivative example in the documentation evaluated a `Dense(2, 1)` network with a 2×2
  weight matrix, which the new linear indexing of parameter arrays would read differently.
- Mismatched batch sizes between two data arguments now raise a `DimensionMismatch` instead of
  silently using the first argument's batch size.
- A matrix-valued equation evaluated on a batch with two batch dimensions and `reduce = hcat` now
  raises an `ArgumentError` explaining the conflict, instead of a bare `DimensionMismatch`.
- A *scalar*-valued equation evaluated on a batch with two batch dimensions and `reduce = hcat` threw
  `ArgumentError: Cannot call tail on an empty tuple`, although the documented shape table lists the
  combination as supported. It returns a ``1\times{}N_1\times{}N_2`` array now, like every other
  ``m = 1`` case.
- A symbolic variable left **free** in an equation — one passed neither as a data variable nor as a
  parameter — and named `k`, `ps`, `out`, `x1` or `x2` was silently bound by the corresponding kernel
  argument: an equation containing `k` evaluated with the *batch index* substituted for it, giving
  wrong numbers with no error and the right answer for the first column. The check that was supposed
  to catch this only inspected the generated argument names, which are always `ˍ₋argN`; it now runs
  on the generated body.
- `Jacobian(f, nn)` threw `MethodError: no method matching vec(::Num)` for a scalar `f`, although the
  documented `vec(f)` row convention covers it. A scalar `f` now gives a ``1\times{}n`` Jacobian.
- An empty batch (`rand(d, 0)`) threw `reducing over an empty collection is not allowed` on the
  out-of-place path, where the in-place path returned an empty result. Both return the empty result
  now.
- Data arguments whose ranks are mixed, or greater than three, raised a `MethodError` naming an
  internal function and the whole `RuntimeGeneratedFunction` type; they now raise an `ArgumentError`
  naming the ranks that were given.
- An equation set whose entry is matrix-valued accepted a batch with two batch dimensions and
  returned an ``(m\cdot{}n)\times{}N_1\times{}N_2`` array, while the same entry built on its own
  threw. Both throw now.
- `docs/Project.toml` pointed `[sources]` at an absolute path on the author's machine, so
  `Pkg.instantiate()` in the documentation environment failed on every other checkout.

### Changed

- **Code generation works on the syntax tree, not on strings**
  ([#44](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/44)). The rewrites used to be
  regexes over the printed form of the code `Symbolics.build_function` emits, hard-coding internal
  Symbolics/SymbolicUtils tokens, with a near-complete duplicate of the pipeline for the
  two-data-argument case. `src/codegen/expression_rewriting.jl` now has one small rule per problem,
  each separately tested, and matches arguments by *position*, which collapses the one- and
  two-argument pipelines into one. `src/build_function/` (≈1300 lines) became `src/codegen/`
  (≈700).
- **Parameters are handed to `Symbolics.build_function` as a flat list of arrays** with their access
  paths recorded (`parameter_arguments`), rather than relying on it to destructure a nested
  `NamedTuple`. The kernel's own interface stays nested.
- **Equation sets containing a scalar entry are built jointly.** They used to fall back to a separate
  per-entry code path, which meant losing the shared forward pass and compiling one
  `RuntimeGeneratedFunction` per entry.
- **`SymbolicPullback` uses two small callable `struct`s** (`ParameterGradient`, `PullbackFunction`)
  instead of nested closures, and no longer needs to unwrap its result
  ([#29](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/29)) — the
  `@warn "There is probably a bug in the code somewhere"` paths are gone.
- Code generation emits 15–20 % less code and batched evaluation is 20–35 % faster, measured with
  `scripts/codegen_comparison.jl`.

### Added

- **A user manual.** `docs/src/` gains a guide (symbolic networks, building functions, derivatives,
  equation sets, training), a `limitations.md` collecting the assumptions and rough edges in one
  place, and `internals/code_generation.md` describing the four stages of the code-generation
  pipeline and why each rewrite rule exists. The README is now a quickstart that links into it.
- **A restructured test suite** ([#39](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/39)):
  unit tests per rewrite rule (covering both the symbol and the function-object form of every
  matched call), the full batching shape matrix
  ({scalar, vector, matrix equation} × {1, 2 data arguments} × {vector, matrix, 3-tensor input} ×
  {`hcat`, `+`} × {in-place, out-of-place}), `@inferred` coverage of every call shape, and a
  codegen-drift guard that asserts the *AST* properties the rewrite rules depend on.
- `AbstractBatchedFunction` and its two concrete subtypes are documented, including a table of
  input shape → result shape.

## Open Issues

Things that came up during this refactor and are **not** fixed.

### Semantics

- **`SymbolicPullback` assumes the loss is additive over the batch.** It differentiates the loss of a
  *single* sample and sums the per-sample gradients (`reduce = +`). That equals the gradient of the
  batched loss only when the loss is a sum over samples.
  `AbstractNeuralNetworks.FeedForwardLoss` is not: it normalises by `norm(output)` taken over the
  whole batch, so for batches of more than one sample the symbolic pullback and a `Zygote` pullback
  of the same `NetworkLoss` disagree. This is pre-existing behaviour — the previous test suite only
  ever exercised batches of one — and cannot be fixed within the current design, which differentiates
  a single symbolic sample. It is documented in `docs/src/limitations.md` and pinned by a test.
- **`FeedForwardLoss` divides by `norm(output)`**, so a target that is identically zero gives
  `NaN`/`Inf`.
- **The element type of an in-place result comes from the inputs, not from the expression**
  (`promoted_eltype`). A `Float32` network whose generated code contains a `Float64` literal rounds
  the literal rather than widening the result. That is the behaviour one wants for a network, but it
  is a deliberate choice rather than a derived one.

### Not differentiable / not supported

- **The default (in-place) result cannot be differentiated by `Zygote`**, because it is produced by
  mutation. `inplace = false` is the escape hatch, at the cost of one allocation per sample. Pinned
  by a test, so the day the default becomes differentiable the keyword can be retired.
- **A matrix-valued equation cannot be evaluated on a batch with two batch dimensions when
  `reduce = hcat`**: concatenating the per-sample results already uses the second dimension. It now
  throws a clear error; supporting it would need a different result layout.
- **All data arguments of a generated function must have the same rank and batch size.**

### Upstream

- [#40](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/40) —
  `test_symbolic_gradient2` remains disabled. The blocker is `AbstractNeuralNetworks.Dense`
  computing `ps.W * x`, which has no method for a three-dimensional `x`; that is the *reference*
  implementation, not this package. A generated function handles such an input fine. Unblocking it
  needs matrix–tensor multiplication upstream (`GeometricMachineLearning` has one). The stale comment
  in the test now states this; the affected case is covered by assembling the reference sample by
  sample instead.
- [#35](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/35) — partially resolved.
  `AbstractNeuralNetworks` 0.6.4 defines `input_dimension`/`output_dimension` for an `AbstractLayer`,
  but not for a `Chain`, which is what this package calls them on. The two `Chain` methods still live
  in `src/symbolic_neuralnet/symbolic_neuralnet.jl` and belong upstream.
- **The rewrite rules depend on undocumented properties of `Symbolics.build_function`'s output** —
  the shape of the emitted function, that data arguments are only read one entry at a time, that
  `create_array` takes a type as its first argument, and that the in-place form addresses its output
  linearly. `test/codegen/codegen_drift.jl` asserts each of them directly so that an upstream change
  fails there with a clear message, but there is no supported interface to rely on instead.
- **`use_base_mapreduce` is no longer triggered by anything this package generates** under
  Symbolics 7; `Symbolics._mapreduce` came from reductions over un-scalarised `Symbolics.Arr`s, which
  the switch to scalar variables rules out. The rule is kept as a `Zygote` safety net for
  user-supplied equations and is covered by a synthetic unit test, but not by any end-to-end one.

### Housekeeping

- `scripts/pullback_comparison.jl` and the untracked `scripts/pullback_comparison_static.jl` depend
  on `GeometricMachineLearning`, which is in no project environment, so neither runs out of the box.
  `pullback_comparison_static.jl` additionally still imports the removed `symbolic_pullback`. See
  also [#9](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/9).
- The generated `api.md` is 106 KiB, above Documenter's 100 KiB warning threshold. Splitting the
  `@autodocs` block per source directory would fix it.
- Remaining open issues untouched by this refactor:
  [#21](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/21) (flatten/destruct/reconstruct),
  [#31](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/31) (convenience wrappers such
  as `parent(jac)` and `build_nn_function(jac, …)`), and
  [#34](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/34) (`ParameterHandling`).
