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
#
# If an upstream release changes how code is generated (renamed arguments, a different
# array constructor, CSE `let`-blocks, …), these regexes silently stop matching and the
# generated functions become wrong or fail to compile. This test asserts the tokens are
# still present at the pipeline stage each rewrite consumes them, so such drift fails here
# with a clear message instead of surfacing as a confusing downstream error.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: _reduce
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Symbolics: @variables
using Test
import Random

Random.seed!(123)

c = Chain(Dense(2, 3, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)

# `fix_create_array` / `rewrite_arguments` / `modify_input_arguments` consume the *raw*
# `build_function` output; `make_kernel` consumes it after a `Meta.parse` round-trip
# (which normalises `(getindex)(x, i)` to `getindex(x, i)`), so we assert on both forms.
@testset "single-input codegen tokens" begin
    eq = Symbolics.scalarize(c(snn.input, params(snn)))
    raw = string(_reduce(build_function(eq, snn.input, values(params(snn))...; expression = Val{true})))
    @test occursin("ˍ₋arg", raw)                            # rewrite_arguments
    @test occursin("SymbolicUtils.Code.create_array", raw)  # fix_create_array
    @test occursin("(sinput, ", raw)                        # modify_input_arguments assertion
    @test occursin(r"getindex\(sinput, [0-9]+\)", string(Meta.parse(raw)))  # make_kernel
end

@testset "double-input codegen tokens" begin
    @variables soutput[1:3]
    eq = Symbolics.scalarize((c(snn.input, params(snn)) - soutput) .^ 2)
    raw = string(_reduce(build_function(eq, snn.input, soutput, values(params(snn))...; expression = Val{true})))
    @test occursin("ˍ₋arg", raw)                            # rewrite_arguments2
    @test occursin("SymbolicUtils.Code.create_array", raw)  # fix_create_array
    @test occursin("(sinput, soutput, ", raw)               # modify_input_arguments2 assertion
    normalised = string(Meta.parse(raw))
    @test occursin(r"getindex\(sinput, [0-9]+\)", normalised)   # make_kernel2
    @test occursin(r"getindex\(soutput, [0-9]+\)", normalised)  # make_kernel2
end

# End-to-end guard: even if the token checks above are satisfied, the assembled function
# must still evaluate correctly over a batch (this exercises the `k`-indexing that
# `make_kernel` injects).
@testset "assembled function is correct over a batch" begin
    f = build_nn_function(c(snn.input, params(snn)), params(snn), snn.input)
    x = rand(2, 5)
    @test f(x, params(nn)) ≈ reduce(hcat, [c(x[:, k], params(nn)) for k in axes(x, 2)])
end
