# Changelog

All notable changes to `SymbolicNeuralNetworks.jl` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] — 2026-08-25

The seam between two layers, which 0.6.0 introduced as a plain vector, becomes something a layer can
widen. That is what
[GML #245](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/245) was waiting for, and
along the way it fixes the promise 0.6.0 made and did not keep. Separately, the walk that splits the
result of a generated function is taken off a closure Julia 1.10 does not always elide, and the suite
gains a gate that measures what a generated function allocates rather than only that it infers.

Resolves [#54](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/54) and
[#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55).

### Fixed

- **An equation set cost 1.85x the allocations to evaluate on Julia 1.10.** Splitting the flat result
  of a jointly generated function back into the nesting of the parameters walks a
  `NeuralNetworkParameters.ParameterLayout`, which 0.6.0 put in place of the local `FlatSlice` of
  0.5.0. Both that walk — `unflatten_batch` here, for a batch — and the `unflatten` it delegates the
  un-batched case to upstream were written as `map` over a closure, which Julia 1.10 does not always
  elide.

  Nothing was type unstable, which is why the suite stayed green: `@inferred` passes on every version,
  and `test/codegen/type_stability.jl` was the only thing measuring this path. What moved was 1056
  bytes per `EquationSetFunction` call on 1.10 against 560 on 1.11 and later — enough that
  `NonlinearIntegrators` 0.4.0 tripped its own allocation gate at 28 096 bytes per Newton residual
  against 15 168 before, and shipped a 1.10-only ceiling to stay green
  ([#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55), reported from
  [NonlinearIntegrators #86](https://github.com/JuliaGNI/NonlinearIntegrators.jl/pull/86)).

  Both walks are `Base.tail` recursion now — the shape `NeuralNetworkParameters` states as its house
  rule and uses for `flatten!`/`unflatten!` already — and the two halves are independent. Bytes per
  call on Julia 1.10, on `Chain(Dense(1, 4, tanh), Dense(4, 1, identity; use_bias = false))`, with
  each half varied on its own:

  | | fixed, with 0.2.1 | fixed, with 0.2.0 | unfixed, with 0.2.1 | unfixed, with 0.2.0 |
  |---|---|---|---|---|
  | `DQDθ`, single sample | **768** | 1056 | **768** | 1056 |
  | `DQDθ`, batch of 8 | **2032** | **2032** | 2320 | 2320 |
  | `split_result`, single sample | **512** | 800 | **512** | 800 |
  | `split_result`, batch of 8 | **1136** | **1136** | 1424 | 1424 |

  So the un-batched rows are `NeuralNetworkParameters`' to move and the batched rows are this
  package's, and neither half moves anything on 1.11, 1.12 or 1.13, where every cell of that table is
  identical. What this walk costs turns out to be a property of the *shape* of the set rather than of
  its depth: allocation is flat in nesting depth and linear in the number of leaves on every version
  measured, and on 1.10 the recursion saves on some shapes — three leaves in two unequal groups, which
  is what a two-layer `Chain` has, go from 640 bytes to 368 — and nothing on others.

  `NonlinearIntegrators` calls `DQDθ` on a length-one `Vector`, so its residual takes the un-batched
  path and it is the upstream half that puts it back at 15 168 on 1.10 and lets the downstream
  ceiling come out. `Project.toml` therefore requires `NeuralNetworkParameters` 0.2.1, which is also
  what the new ceilings in `test/codegen/allocations.jl` are reachable against.

  `split_result`, the `unflatten_batch` methods and `_unflatten_batch_children` are marked `@inline`
  alongside. That did not move the measurement — it is consistency with the rule the upstream walks
  already follow, not a claimed effect. The rewrite does take the walk off `Base`'s `Any32` fallback,
  which `map` drops to past 32 children and which returns a tuple with no concrete type.

  The report suspected `symbolic_parameter_gradient` and named `DQDθ`, `DVDθ` and `V_func`. It is the
  first two: both are equation sets, and `V_func` is a single equation that does not go through this
  path at all — and is not called by the affected integrators.

- **`layerwise = :auto` declined instead of throwing when a layer cannot be seeded.** 0.6.0 said
  "`:auto` never raises where the monolithic construction would have built; only `layerwise = true`
  does", and `29f1a9e` had made that true of the *loss*. It was not true of the *layers*.
  `symbolic_steps` asks whether the model decomposes and whether each layer has known dimensions;
  nothing asked whether a layer can be given the fresh variables the construction puts at the seams.
  `GeometricMachineLearning`'s `GeneralizedHamiltonianArchitecture` composes `SymplecticEuler` layers
  that thread `(state, system parameters)` from layer to layer, so all but the last return a `Tuple`;
  `composes_layerwise` said yes and `layer_seed` then threw
  `MethodError: no method matching scalar_expressions(::Tuple{…})`, on a network `layerwise = false`
  builds in 3.2 s.

  `checked_layer_seed` is the layer-side counterpart of `checked_guess` and draws its `try` in the
  same place — around building the seed and nothing else. It catches rather than testing
  `applicable(scalar_expressions, layer(sx, ps))`, because a layer can fail to be seeded in two ways
  and that predicate only covers one: the layer *downstream* of a tuple-returning one has no method
  for a bare vector at all, and has to have been applied before `applicable` can be asked anything.

  `layerwise = true` still raises, and now names which of the three things stood in the way — the
  offending layers by key and type, where that is the reason — rather than listing two of them and
  leaving the reader to choose. `layerwise_gradient_function` gained a `demanded` keyword and both
  outcomes go through `decline`, so there is one decision path rather than a second traversal that
  could drift from it.

- **`loss_expression` was missing from the manual**, so every `@ref` to it failed to resolve and took
  the documentation build down with them. A comment sat between the docstring and the definition, and
  Julia does not attach a docstring across one, so the binding carried no documentation at all and
  `@autodocs` emitted nothing for it. Introduced in 0.6.0, and surfacing only now: the docs
  environment could not resolve until `GeometricMachineLearning` 0.6 was registered, so the build
  never reached its cross-references.

- **The API page was 2 KiB from taking the documentation build down.** `api.md` is one `@autodocs`
  block over the whole package, so it grows with every docstring, and it had reached 198 KiB against
  Documenter's `size_threshold` default of 200 KiB — a hard error rather than a warning. Raised to
  512 KiB in `docs/make.jl`; `size_threshold_warn` is left at its default, so the page still says it
  is large.

### Added

- **The suite measures what a generated function allocates**, in `test/codegen/allocations.jl`, with
  the same ceilings on every Julia version. It pinned inference and nothing else before, which is how
  #55 reached a dependent package's release rather than CI, on a matrix that already runs 1.10. The
  same file asserts that the batched walk stays inferable past 32 children, which is the point `map`
  used to drop to `Base`'s `Any32` fallback. `scripts/allocation_comparison.jl` prints the per-layer
  figures the ceilings are set from and takes a call apart — `promoted_eltype`, a bare kernel, an
  equation set, and the splitting on its own, single sample against batch — so that a regression can
  be attributed rather than just noticed.

- **A layer can declare how it meets the seam**, with four functions that each default to the
  plain-vector construction, so nothing that worked before is affected:
  `carried_variables`, `seam_value`, `state_expressions` and `seam_arguments`. Declared together, they
  let the layerwise pullback compose a chain whose layers pass data on alongside the state. See
  `seam_interface`, and the "Layers" section of the training guide.

  What a layer carries is *data*, never a differentiation target: λ pairs with the state, the
  seed is differentiated with respect to the state and the layer's parameters, and the carried
  variables become further arguments of the two generated kernels. So a carrying layer costs the same
  two calls per sweep as any other.

  This matters for correctness and not only for build time. The monolithic construction traces the
  chain from a plain vector, so a layer that *defaults* what it carries — `SymplecticEuler` defaults
  the parameters of the system to `NullParameters` — is differentiated with that default whatever the
  caller passes. Before this there was no construction that could be told otherwise.

  `carried_variables` has to name its arrays something of its own: `layer_seed` calls the state `x`
  and the sensitivities `λ`, and reusing either name gives one symbolic array two argument slots
  rather than declaring a second array. `Symbolics.build_function` then makes the generated code read
  both slots from the last one, so the kernels build, run and return a wrong gradient — so
  `layer_step` rejects such a seam outright rather than declining, a decline being what a *layer*
  that cannot be seeded deserves and not a layer whose declaration is wrong.

- **`SymbolicPullback`'s input may be a `Tuple`**, so that a model whose layers thread a pair can be
  handed one. The output half stays an array or a `NamedTuple`: it is the target the seed compares the
  network's output to. Two methods rather than one wider signature, because
  `AbstractNeuralNetworks` defines the fallback on `AbstractPullback` and
  `Tuple{<:ArrayOrNamedTuple, <:ArrayOrNamedTuple}`, and a single method more specific in its first
  argument and less specific in its third would be an ambiguity rather than an override.

### Changed

- **A generated function may take any number of data arguments**, where two was the cap. The plumbing
  was already variadic — `build_nn_function`, `AbstractBatchedFunction{NDATA}`, the rewrite rules —
  and the limit was the fixed `DATA_NAMES = (:x1, :x2)`. Names are now derived on demand
  (`data_name(i)`), and `AbstractBatchedFunction`, `EquationSetFunction` and
  `EquationSetArrayFunction` gained a general-arity call in addition to the one- and two-argument ones
  they had. A layerwise sweep over a carrying layer needs three.

  With that, the *whole* `x1`, `x2`, … family is reserved rather than only the arities in use
  (`is_reserved_name`, replacing `RESERVED_NAMES`): a symbolic variable left free in an equation and
  named `x3` used to pass the check, and would have broken silently the day a third data argument
  arrived.

### Breaking

- `layer_seed` returns `(seed, sparams, sdata, sλ)`, where `sdata` is the *tuple* of the seam's data
  variables — the state first — in place of the single `sx` it returned before. `layer_step` takes
  that tuple rather than a parameter prototype, so that a chain can be seeded, and declined, before
  any code is generated for any of its layers. Both are internal; `scripts/codegen_comparison.jl` is
  the only caller outside the construction, and the node counts it reports are unchanged.
- `SymbolicNeuralNetworks.RESERVED_NAMES` and `DATA_NAMES` are gone, replaced by `is_reserved_name`,
  `data_name` and `FIXED_NAMES`.

## [0.6.0] — 2026-08-24

The rest of the 0.5.0 refactor, and the reason it is breaking: the package follows
`AbstractNeuralNetworks` 0.7 off its own parameter container. Alongside that, `SymbolicPullback`
stops building an expression that grows exponentially in the depth of the network, and the flat
parameter vector [#21](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/21) asked for
arrives. Nothing in the exported surface — `SymbolicNeuralNetwork`,
`AbstractSymbolicNeuralNetwork`, `build_nn_function`, `SymbolicPullback` — is renamed or removed;
`build_flat_function` and `flat_parameter_gradient` are added to it.

Resolves [#21](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/21) and
[#49](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/49).

### Breaking

- **The parameter container comes from `NeuralNetworkParameters`**, and the compat bound moves to
  `AbstractNeuralNetworks = "0.7"` (was `"0.6.4"`). 0.7 moved the container out to its own package
  and removed the old name outright rather than leaving an alias behind, so that one type has one
  name across the ecosystem: `AbstractNeuralNetworks.NeuralNetworkParameters` is now
  `NeuralNetworkParameters.NetworkParameters`. It is the same type object, so code that only *uses*
  a parameter set is unaffected; code that *names* the type has to be updated. Every call site here
  names it where it now lives.

  The import is selective rather than a bare `using`, because `NeuralNetworkParameters` exports
  `flatten` and `unflatten`, and `unflatten` in this package is a different concept — laying a flat
  vector of generated values back into the shape of an *equation* set. A sweep of all 23 upstream
  exports found that to be the only collision.

- **`QPTOAT` is gone, replaced by `AbstractNeuralNetworks.ArrayOrNamedTuple`**
  ([AbstractNeuralNetworks #31](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/issues/31)).
  0.6's alias was `Union{NamedTuple{(:q, :p), Tuple{AT, AT}}, AbstractArray}`; 0.7 dropped it on the
  grounds that Hamiltonian phase-space vocabulary has no business in an architecture-agnostic
  package, and replaced it with a key-agnostic one. The new alias is strictly wider, so
  `SymbolicPullback`'s input–output method accepts everything it used to and no longer refuses a
  `NamedTuple` that is not keyed `(:q, :p)`.

### Changed

- **Equation sets are laid out by `NeuralNetworkParameters.ParameterLayout`** rather than by a local
  `FlatSlice`. Flattening a nested collection of symbolic equations into one vector is the same
  problem as flattening a parameter set, because an equation set has the same shape as one, so
  `flatten_equations` is now `flatten` over the layout upstream builds. `unflatten_batch` stays here:
  `unflatten` upstream already has a matrix method, and it means the other thing a matrix can mean —
  splitting the rows of a Jacobian, with no batch dimension to restore. `symbolic_differentials`,
  `symbolic_derivative` and `promoted_eltype` likewise use `mapparameters` and `parameter_eltype`
  instead of each carrying its own recursion over a parameter set.
- **The docs environment requires `GeometricMachineLearning = "0.6"`.** 0.6 is the first GML that
  tracks `AbstractNeuralNetworks` 0.7, and the first whose single generic `optimization_step!` takes
  a `NetworkParameters` gradient directly — which is what let the compatibility shim in
  `docs/make.jl` go away. Stating the bound means the docs environment fails to resolve rather than
  quietly resolving against a GML that predates the generic method.

### Added

- **`SymbolicPullback` is composed layer by layer**
  ([#49](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/49)). It used to build one
  scalar expression for the loss of the whole network and differentiate it once per scalar parameter.
  A `Chain`'s forward pass is inlined layer into layer, so that expression is
  `O(width^depth)` before anything is differentiated, and differentiating it walks the whole of it
  once per parameter: four layers of width 16 — 626 parameters — reach a gradient expression of 2·10⁸
  nodes and never build. `cse` cannot help, as it runs on an expression that has already been built.

  Each layer now gets fresh symbolic variables for its own input, and per layer the *scalar*
  `λₖ · fₖ(xₖ₋₁; θₖ)` is differentiated twice — once with respect to the layer's input, which gives
  `λₖ₋₁`, and once with respect to its parameters, which gives `∂L/∂θₖ`. The composition happens when
  the pullback is evaluated. The symbolic material becomes a sum over layers rather than a product:
  2 520 nodes instead of 388 700 at four layers of width 4, 68 760 instead of 209 455 964 at width 16,
  and exactly 864 more per identical added layer. The network that did not build now builds in half a
  second.

  `SymbolicPullback`'s signature, its functor and its return type are unchanged. The new `layerwise`
  keyword selects the construction; `:auto`, the default, composes layer by layer for every model that
  decomposes into more than one layer. `layerwise = false` recovers the previous construction, which
  is also what a model that does not decompose into layers falls back to — as does a loss the
  layerwise construction cannot get a seed from, whether because the guessed expression disagrees
  with the loss or because the loss cannot be applied to a `PassThroughLayer` at all. `:auto` never
  raises where the monolithic construction would have built; only `layerwise = true` does.

  This also fixes the second-derivative case of
  [GML #245](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/245), where a layer
  itself *contains* a built symbolic gradient: with a seam at each layer the layer is called rather
  than traced, so stacking two costs a function call instead of inlining a gradient expression inside
  a gradient expression.

- **`loss_expression`, an extension point for the loss** — the layerwise construction needs the loss
  as a function of the network's *prediction*, which `AbstractNeuralNetworks` has no interface for.
  By default it is obtained by applying the loss to a `PassThroughLayer`, a model whose prediction is
  its input. That is right for a loss which reaches its model once and compares the result to
  `output`, and wrong for one that does something else — an autoencoder loss compares the prediction
  to the network's *input*, and so reads through a pass-through model as identically zero. The guess
  is therefore checked against the loss itself before it is used, and the construction falls back to
  the monolithic one when the two disagree, rather than returning a zero gradient. A loss can declare
  its expression instead, in which case it is used as given.

- **Generated functions that take their parameters flat**
  ([#21](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/21)), both **exported**.
  `build_flat_function` is `build_nn_function` with a flat vector in place of the parameter set, and
  `flat_parameter_gradient` differentiates with respect to the parameters and lays the result out
  flat — a vector for a scalar expression, and for an array-valued one the `length(f) × flatlength`
  Jacobian a Newton step is built from. Out of place, so a `Dual`-valued vector gives `Dual`-valued
  parameters and `ForwardDiff` differentiates with respect to the flat form.

  Neither reads a model: both take a `NetworkParameters` of symbolic leaves in place of the symbolic
  network, which is what the issue's second half — degrees of freedom of a nonlinear expression that
  is not a network's forward pass — goes through. `symbolic_parameter_gradient` gained the same
  method. Both gradient functions take any `EquationSet`, so a nested `NamedTuple` of symbolic leaves
  works as well — which is also what the parameters of a `SymbolicNeuralNetwork` are allowed to be.
  `build_flat_function` is the narrower of the two and wants a `NetworkParameters`, because
  `build_nn_function` dispatches on one.

  The conversion itself is not reimplemented here: `NeuralNetworkParameters` has `flatten`/`unflatten`
  over a reusable `ParameterLayout`, allocation-free variants, `Float32` fidelity, a GPU-safe forward
  conversion (`copyto!` per leaf, no element ever indexed) and `ChainRulesCore` rules for both
  directions — though its reverse rule accumulates cotangents elementwise, so reverse mode *through*
  the conversion is not itself GPU-clean. The new `docs/src/guide/flat_parameters.md` says which half
  is whose.

## [0.5.0] — 2026-08-14

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

Things that came up during the 0.5/0.6 refactor and are **not** fixed.

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
- **All data arguments of a generated function must have the same rank and batch size.** For a layer
  that carries data alongside the state (`seam_interface`) that is a constraint on what
  `seam_arguments` returns: carried data that is the same for the whole batch has to be broadcast out
  to one column per sample, and a carried datum that varies per sample cannot be combined with a state
  that has two batch dimensions.
- **A layer that carries data alongside the state cannot be the last one in a chain**, since the
  chain's output is what the loss and its seed compare against the target.

### Upstream

- **`docs/Project.toml` does not resolve.** `Project.toml` requires `NeuralNetworkParameters` 0.2.1,
  because that is the release whose `unflatten` the ceilings in `test/codegen/allocations.jl` are
  reachable against; the released `GeometricMachineLearning` 0.6.0, which the docs build the training
  guide against, pins `NeuralNetworkParameters = "0.1"`:

  ```
  ERROR: Unsatisfiable requirements detected for package GeometricMachineLearning [194d25b2]:
   ├─restricted to versions 0.6 by project, leaving only versions: 0.6.0
   └─restricted by compatibility requirements with NeuralNetworkParameters [67f4d93a] to versions:
     0.1.0 - 0.5.0 or uninstalled — no versions left
  ```

  The Documentation job stays red until a `GeometricMachineLearning` release tracks the 0.2
  container. It is blocked at the 0.7.0 version bump regardless — GML 0.6.0 also pins
  `SymbolicNeuralNetworks = "0.6"` — so the two unblock together.
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
- One open issue is untouched by this refactor:
  [#31](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/31) (convenience wrappers such
  as `parent(jac)` and `build_nn_function(jac, …)`).
