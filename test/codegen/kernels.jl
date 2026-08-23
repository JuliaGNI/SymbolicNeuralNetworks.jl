# The kernels are what the rewrite rules are applied to. They evaluate *one* sample of a batch and
# are the layer below `AbstractBatchedFunction`, which adds the batching itself.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: build_kernel, build_kernel!, parameter_arguments, generated_expression,
                              _assert_no_name_clash, _assert_no_reserved_names_in_body
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Test
import Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
eq = c(snn.input, params(snn))

@testset "parameter_arguments flattens the parameter tree" begin
    paths, arrays = parameter_arguments(params(snn))
    @test paths == ((:L1, :W), (:L1, :b), (:L2, :W), (:L2, :b))
    @test all(map(===, arrays, (params(snn).L1.W, params(snn).L1.b, params(snn).L2.W, params(snn).L2.b)))
    # every parameter array has to become an argument of its own: `Symbolics.build_function` only
    # recognises a symbolic array it is handed directly, and leaves the entries of anything else as
    # free variables in the generated code
    @test length(arrays) == 4
end

@testset "the out-of-place kernel evaluates one sample of a batch" begin
    kernel = build_kernel(Symbolics.scalarize(eq), params(snn), snn.input)
    input = rand(3, 5)
    @test all(k -> kernel(input, ps, k) ≈ c(input[:, k], ps), axes(input, 2))
end

@testset "the in-place kernel writes sample k where reduce = $reduction puts it" for reduction in (hcat, +)
    kernel! = build_kernel!(Symbolics.scalarize(eq), params(snn), snn.input; reduction = reduction)
    input = rand(3, 5)
    reference = reduce(hcat, [c(input[:, k], ps) for k in axes(input, 2)])

    if reduction === hcat
        out = zeros(2, 5)
        for k in axes(input, 2)
            kernel!(out, input, ps, k)
        end
        @test out ≈ reference
    else
        out = zeros(2)
        for k in axes(input, 2)
            kernel!(out, input, ps, k)
        end
        @test out ≈ vec(sum(reference; dims = 2))
    end
end

@testset "two data arguments" begin
    soutput = Symbolics.variables(:y, 1:2)
    two_input_eq = Symbolics.scalarize((c(snn.input, params(snn)) - soutput) .^ 2)
    kernel = build_kernel(two_input_eq, params(snn), snn.input, soutput)
    input, output = rand(3, 4), rand(2, 4)
    @test all(k -> kernel(input, output, ps, k) ≈ (c(input[:, k], ps) - output[:, k]) .^ 2, axes(input, 2))
end

# `Symbolics.build_function` emits no in-place form for a scalar equation.
@testset "a scalar equation has no in-place kernel" begin
    scalar_eq = sum(c(snn.input, params(snn)))
    @test isnothing(build_kernel!(scalar_eq, params(snn), snn.input; reduction = hcat))
    @test isnothing(generated_expression(scalar_eq, (snn.input,), last(parameter_arguments(params(snn)));
                                         inplace = true, cse = true))
    kernel = build_kernel(scalar_eq, params(snn), snn.input)
    input = rand(3, 4)
    @test all(k -> kernel(input, ps, k) ≈ sum(c(input[:, k], ps)), axes(input, 2))
end

# The rewrite rules identify arguments by name, so a symbolic variable carrying one of the names the
# kernel uses for itself has to be rejected rather than silently rewritten twice.
@testset "reserved argument names are rejected" begin
    @test_throws ArgumentError _assert_no_name_clash([:ˍ₋arg1, :ps])
    @test_throws ArgumentError _assert_no_name_clash([:k])
    @test isnothing(_assert_no_name_clash([:ˍ₋arg1, :ˍ₋arg2, :W_5]))
end

# A variable that is passed to `Symbolics.build_function` becomes an argument and is renamed to
# `ˍ₋argN`; one that is left *free* in the equation survives into the body under its own name. If
# that name is `k` it used to be bound by the kernel's batch index, so the equation evaluated with
# the column number in place of the variable — no error, wrong numbers, right answer for column 1.
@testset "a free symbolic variable named like a kernel argument is rejected" begin
    for name in (:k, :ps, :out, :x1, :x2)
        free = Symbolics.variable(name)
        @test_throws ArgumentError build_kernel(free .* eq, params(snn), snn.input)
        @test_throws ArgumentError build_kernel!(free .* eq, params(snn), snn.input; reduction = hcat)
        @test_throws ArgumentError build_nn_function(free .* eq, snn)
    end
end

@testset "_assert_no_reserved_names_in_body" begin
    @test isnothing(_assert_no_reserved_names_in_body(:((*)(ˍ₋arg1[1], ˍ₋arg2[1]))))
    @test isnothing(_assert_no_reserved_names_in_body(Expr(:call, Base.getindex, :ˍ₋arg1, 1)))
    @test_throws ArgumentError _assert_no_reserved_names_in_body(:((*)(k, ˍ₋arg1[1])))
    @test_throws ArgumentError _assert_no_reserved_names_in_body(Expr(:call, Base.:*, :ps, 1))
end

# ... whereas a *data* variable of that name is fine: it becomes an argument and gets renamed.
@testset "a data variable may be named like a kernel argument" begin
    variables = Symbolics.variables(:k, 1:3)
    f = build_nn_function(c(variables, params(snn)), params(snn), variables)
    input = rand(3, 4)
    @test f(input, ps) ≈ reduce(hcat, [c(input[:, i], ps) for i in axes(input, 2)])
end

@testset "at most two data arguments" begin
    third = Symbolics.variables(:z, 1:2)
    soutput = Symbolics.variables(:y, 1:2)
    @test_throws ArgumentError build_kernel(Symbolics.scalarize(eq), params(snn), snn.input, soutput, third)
end
