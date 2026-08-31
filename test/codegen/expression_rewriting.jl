# Unit tests for the rewrite rules in `src/codegen/expression_rewriting.jl`.
#
# Each rule is exercised on both forms in which `SymbolicUtils` refers to a function: by symbol and
# by embedding the function *object* in the tree. A rule that matches only the symbol silently does
# nothing on real generated code — and, in the case of `index_by_batch`, still produces a result
# that is correct for the first sample of a batch, which makes the bug invisible in a smoke test.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: postwalk, callee_name, function_arguments_and_body,
                              function_expression,
                              argument_substitutions, access_expression, substitute_symbols,
                              use_generic_array_constructor, use_base_mapreduce,
                              index_by_batch,
                              accumulate_into_output
using Test
# `SymbolicUtils` is reached through the package rather than imported, so that the test suite does
# not need it as a dependency of its own.
using SymbolicNeuralNetworks: SymbolicUtils

@testset "callee_name recognises every form" begin
    @test callee_name(:(getindex(x, 1))) === :getindex
    @test callee_name(Expr(:call, Base.getindex, :x, 1)) === :getindex
    @test callee_name(:(SymbolicUtils.Code.create_array(Array))) === :create_array
    @test callee_name(Expr(:call, GlobalRef(Base, :sin), :x)) === :sin
    @test callee_name(:(x[1])) === nothing
    @test callee_name(:x) === nothing
end

@testset "postwalk visits children first and does not revisit results" begin
    # `f` would loop forever if the symbol it inserts were visited again
    @test postwalk(node -> node === :a ? :(a + 1) : node, :(a * 2)) == :((a + 1) * 2)
end

@testset "function_arguments_and_body" begin
    expr = :(function (u, v)
        ;u + v;
    end)
    names, body = function_arguments_and_body(expr)
    @test names == [:u, :v]
    # round-trip: renaming the arguments is all `function_expression` has to do
    @test function_expression((:a, :b), body) == Expr(:function, :((a, b)), body)
    @test function_arguments_and_body(function_expression((:a, :b), body))[1] == [:a, :b]

    @test_throws ArgumentError function_arguments_and_body(:(x + 1))
    @test_throws ArgumentError function_arguments_and_body(:(function f(u)
        ;u;
    end))
end

@testset "argument_substitutions maps arguments by position" begin
    substitutions = argument_substitutions(
        [:a, :b, :c, :d], (:x1, :x2), ((:L1, :W), (:L1, :b));
        output_name = nothing)
    @test substitutions[:a] == :x1
    @test substitutions[:b] == :x2
    @test substitutions[:c] == :(ps.L1.W)
    @test substitutions[:d] == :(ps.L1.b)

    # the in-place form prepends an output argument, which shifts everything else along
    with_output = argument_substitutions([:o, :a, :c], (:x1,), ((:L1, :W),); output_name = :out)
    @test with_output[:o] == :out
    @test with_output[:a] == :x1
    @test with_output[:c] == :(ps.L1.W)

    @test_throws ArgumentError argument_substitutions([:a], (:x1,), ((:L1,),); output_name = nothing)
end

@testset "access_expression" begin
    @test access_expression(:ps, ()) == :ps
    @test access_expression(:ps, (:L1,)) == :(ps.L1)
    @test access_expression(:ps, (:L1, :W)) == :(ps.L1.W)
end

@testset "substitute_symbols" begin
    @test substitute_symbols(:(a + b), Dict{Symbol, Any}(:a => :(ps.L1))) == :(ps.L1 + b)
    # the substituted expression must not be substituted into again
    @test substitute_symbols(:a, Dict{Symbol, Any}(:a => :(ps.a), :ps => :nope)) == :(ps.a)
end

@testset "use_generic_array_constructor ($form)" for (form, expr) in ("symbol callee" =>
    :((SymbolicUtils.Code.create_array)(typeof(ˍ₋arg1), nothing, Val{1}(), a)),
    "function callee" => Expr(:call, SymbolicUtils.Code.create_array,
    Expr(:call, :typeof, :ˍ₋arg1), :nothing, :(Val{1}()), :a))
    rewritten = use_generic_array_constructor(expr)
    @test rewritten.args[2] === :Array
    @test rewritten.args[3:end] == expr.args[3:end]
    @test rewritten.args[1] == expr.args[1]   # the callee is left alone
end

@testset "use_generic_array_constructor leaves other calls alone" begin
    @test use_generic_array_constructor(:(f(typeof(x), y))) == :(f(typeof(x), y))
    @test use_generic_array_constructor(:(create_array(Array, y))) ==
          :(create_array(Array, y))
end

@testset "use_base_mapreduce" begin
    @test use_base_mapreduce(:(Symbolics._mapreduce(
        identity, +, x, Colon(), (:init => false,)))) ==
          :(mapreduce(identity, +, x; dims = Colon()))
    @test use_base_mapreduce(:(mapreduce(identity, +, x))) == :(mapreduce(identity, +, x))
end

@testset "index_by_batch ($form)" for (form, expr, expected) in [
    ("ref form", :(x1[1] + x1[2]), :(x1[1, k] + x1[2, k])),
    ("getindex symbol", :(getindex(x1, 1)), :(getindex(x1, 1, k))),
    ("getindex object", Expr(:call, Base.getindex, :x1, 1),
        Expr(:call, Base.getindex, :x1, 1, :k))]
    @test index_by_batch(expr, (:x1,)) == expected
end

@testset "index_by_batch only touches the data arguments" begin
    # parameters are read the same way but must keep their single index
    @test index_by_batch(:(x1[1] * ps.L1.W[2]), (:x1,)) == :(x1[1, k] * ps.L1.W[2])
    @test index_by_batch(:(x1[1] + x2[1]), (:x1, :x2)) == :(x1[1, k] + x2[1, k])
    @test index_by_batch(:(x1[1] + x2[1]), (:x1,)) == :(x1[1, k] + x2[1])
end

@testset "index_by_batch rejects a data argument used as a whole" begin
    @test_throws ArgumentError index_by_batch(:(sum(x1)), (:x1,))
    @test_throws ArgumentError index_by_batch(:(A[x1]), (:x1,))
end

@testset "accumulate_into_output, reduce = hcat" begin
    @test accumulate_into_output(:(out[1] = a), :out, hcat, 1) ==
          :(out[1 + (k - 1) * 1] = a)
    body = Expr(:block, :(out[1] = a), :(out[2] = b))
    @test accumulate_into_output(body, :out, hcat, 2) ==
          Expr(:block, :(out[1 + (k - 1) * 2] = a), :(out[2 + (k - 1) * 2] = b))
end

@testset "accumulate_into_output, reduce = +" begin
    body = Expr(:block, :(out[1] = a), :(out[2] = b))
    @test accumulate_into_output(body, :out, +, 2) ==
          Expr(:block, :(out[1] += a), :(out[2] += b))
end

# Without this check a rule that stopped matching would leave a kernel that still compiles and runs,
# but writes every sample of the batch into the same place.
@testset "accumulate_into_output insists on one write per equation entry" begin
    body = Expr(:block, :(out[1] = a), :(out[2] = b))
    @test_throws ArgumentError accumulate_into_output(body, :out, hcat, 3)
    @test_throws ArgumentError accumulate_into_output(body, :out, hcat, 1)
    @test_throws ArgumentError accumulate_into_output(:(result[1] = a), :out, hcat, 1)
end
