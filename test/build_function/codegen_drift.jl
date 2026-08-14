# Regression guard against Symbolics / SymbolicUtils code-generation drift.
#
# `build_nn_function` (see `src/build_function/`) does not manipulate the symbolic
# expression tree directly: it rewrites the *string* of the code that
# `Symbolics.build_function` emits. Those rewrites hard-code tokens that are internal
# implementation details of Symbolics/SymbolicUtils:
#
#   * the `ˍ₋argN` parameter names                     -> `rewrite_arguments`
#   * the `SymbolicUtils.Code.create_array` constructor -> `fix_create_array`
#   * the `(sinput, …)` / `(sinput, soutput, …)` signature -> `modify_input_arguments[2]`
#   * the `getindex(sinput, i)` / `getindex(soutput, i)` accessors -> `make_kernel[2]`
#   * the `(ˍ₋out, sinput, …)` signature                -> `modify_input_arguments_iip[2]`
#   * the *linearly indexed* `ˍ₋out[i] = …` writes      -> `make_kernel_iip[2]`
#
# If an upstream release changes how code is generated (renamed arguments, a different
# array constructor, …), these regexes silently stop matching and the generated functions
# become wrong or fail to compile. This test asserts the tokens are still present at the
# pipeline stage each rewrite consumes them, so such drift fails here with a clear message
# instead of surfacing as a confusing downstream error.
#
# Both code-generation modes are checked. With `cse = true` (the default, see
# `_build_nn_function`) the body becomes a `let` block of `var"##cse#N"` bindings; the
# rewrites must survive that too. In particular Symbolics declines to hoist `getindex` of
# `@variables`-created arrays into CSE bindings, which is what keeps `make_kernel` working.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: _reduce, _reduce_iip
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Symbolics: @variables
using Test
import Random

Random.seed!(123)

# Two layers: a single `Dense` has no repeated subexpressions, so CSE would not kick in and
# the `##cse#` assertion below would be vacuous.
c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)

# `fix_create_array` / `rewrite_arguments` / `modify_input_arguments` consume the *raw*
# `build_function` output; `make_kernel` consumes it after a `Meta.parse` round-trip
# (which normalises `(getindex)(x, i)` to `getindex(x, i)`), so we assert on both forms.
@testset "single-input codegen tokens (cse = $cse)" for cse in (false, true)
    eq = Symbolics.scalarize(c(snn.input, params(snn)))
    raw = string(_reduce(build_function(eq, snn.input, values(params(snn))...; expression = Val{true}, cse = cse)))
    @test occursin("ˍ₋arg", raw)                            # rewrite_arguments
    @test occursin("SymbolicUtils.Code.create_array", raw)  # fix_create_array
    @test occursin("(sinput, ", raw)                        # modify_input_arguments assertion
    @test occursin(r"getindex\(sinput, [0-9]+\)", string(Meta.parse(raw)))  # make_kernel
    # CSE must actually have happened, otherwise the assertions above are just a re-run of
    # the `cse = false` case and no longer guard the `let`-block form.
    @test occursin("##cse#", raw) == cse
end

@testset "double-input codegen tokens (cse = $cse)" for cse in (false, true)
    @variables soutput[1:2]
    eq = Symbolics.scalarize((c(snn.input, params(snn)) - soutput) .^ 2)
    raw = string(_reduce(build_function(eq, snn.input, soutput, values(params(snn))...; expression = Val{true}, cse = cse)))
    @test occursin("ˍ₋arg", raw)                            # rewrite_arguments2
    @test occursin("SymbolicUtils.Code.create_array", raw)  # fix_create_array
    @test occursin("(sinput, soutput, ", raw)               # modify_input_arguments2 assertion
    normalised = string(Meta.parse(raw))
    @test occursin(r"getindex\(sinput, [0-9]+\)", normalised)   # make_kernel2
    @test occursin(r"getindex\(soutput, [0-9]+\)", normalised)  # make_kernel2
    @test occursin("##cse#", raw) == cse
end

# The in-place half of `build_function`'s output is what `build_nn_function` actually runs.
# Two properties of it are load-bearing and neither is documented upstream:
#   * `ˍ₋out` is *not* counted in the `ˍ₋argN` numbering, so `rewrite_arguments` is reused as-is;
#   * the output is addressed with a single *linear* index whatever the shape of the equation,
#     which is what lets `make_kernel_iip` offset the writes by the batch index instead of
#     handing the kernel a view.
@testset "in-place codegen tokens (cse = $cse)" for cse in (false, true)
    vector_valued = Symbolics.scalarize(c(snn.input, params(snn)))
    matrix_valued = Symbolics.scalarize(SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(snn)))
    @test ndims(matrix_valued) == 2  # otherwise the linear-indexing check below is vacuous

    for eq in (vector_valued, matrix_valued)
        raw = string(_reduce_iip(build_function(eq, snn.input, values(params(snn))...; expression = Val{true}, cse = cse)))
        @test occursin("(ˍ₋out, sinput, ", raw)                 # modify_input_arguments_iip assertion
        @test occursin("ˍ₋arg2", raw)                           # rewrite_arguments: numbering starts at 2, not 3
        @test occursin(r"getindex\(sinput, [0-9]+\)", string(Meta.parse(raw)))  # make_kernel_iip
        @test occursin(r"ˍ₋out\[[0-9]+\] = ", string(Meta.parse(raw)))          # redirect_output_writes
        @test !occursin(r"ˍ₋out\[[0-9]+, [0-9]+\] = ", string(Meta.parse(raw))) # ... linear, never cartesian
    end
end

@testset "in-place double-input codegen tokens (cse = $cse)" for cse in (false, true)
    @variables soutput[1:2]
    eq = Symbolics.scalarize((c(snn.input, params(snn)) - soutput) .^ 2)
    raw = string(_reduce_iip(build_function(eq, snn.input, soutput, values(params(snn))...; expression = Val{true}, cse = cse)))
    @test occursin("(ˍ₋out, sinput, soutput, ", raw)        # modify_input_arguments_iip2 assertion
    @test occursin("ˍ₋arg3", raw)                           # rewrite_arguments2: numbering starts at 3, not 4
    normalised = string(Meta.parse(raw))
    @test occursin(r"getindex\(sinput, [0-9]+\)", normalised)   # make_kernel_iip2
    @test occursin(r"getindex\(soutput, [0-9]+\)", normalised)  # make_kernel_iip2
    @test occursin(r"ˍ₋out\[[0-9]+\] = ", normalised)           # redirect_output_writes
end

# End-to-end guard: even if the token checks above are satisfied, the assembled function
# must still evaluate correctly over a batch (this exercises the `k`-indexing that
# `make_kernel` injects).
@testset "assembled function is correct over a batch (cse = $cse)" for cse in (false, true)
    f = build_nn_function(c(snn.input, params(snn)), params(snn), snn.input; cse = cse)
    x = rand(2, 5)
    @test f(x, params(nn)) ≈ reduce(hcat, [c(x[:, k], params(nn)) for k in axes(x, 2)])
end
