# Regression guard against Symbolics / SymbolicUtils code-generation drift.
#
# The rewrite rules in `src/codegen/expression_rewriting.jl` work on the syntax tree rather than on
# its printed form, but they still rely on properties of that tree which are implementation details
# of Symbolics/SymbolicUtils and are not documented upstream:
#
#   * the emitted code is an anonymous `function` with one argument per symbolic array it was given,
#     in the order they were given                          -> argument_substitutions
#   * a data argument is only ever read one entry at a time  -> index_by_batch
#   * the array constructor takes the array type as its first argument, filled in with the type of
#     one of the arguments                                   -> use_generic_array_constructor
#   * the in-place form prepends its output argument, and addresses it with a single *linear* index
#     whatever the shape of the equation                     -> accumulate_into_output
#
# If an upstream release changes any of that, the rules would stop matching. Most of them then throw
# by themselves; this test asserts the properties directly, so that drift fails here with a clear
# message rather than surfacing as a confusing downstream error. Both code-generation modes are
# checked: with `cse = true` (the default) the body becomes a `let` block of `var"##cse#N"` bindings,
# which the rules must survive too.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative, generated_expression,
                              parameter_arguments,
                              function_arguments_and_body, callee_name, postwalk,
                              use_generic_array_constructor
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Test
import Random

Random.seed!(123)

# Two layers: a single `Dense` has no repeated subexpressions, so CSE would not kick in and the
# `##cse#` assertion below would be vacuous.
c = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
soutput = Symbolics.variables(:y, 1:2)

"Collect every node of `expr` for which `predicate` holds."
function collect_nodes(predicate, expr)
    found = []
    postwalk(expr) do node
        predicate(node) && push!(found, node)
        node
    end
    found
end

"Every single-index read `name[i]` / `getindex(name, i)` in `expr`."
function reads_of(expr, name::Symbol)
    collect_nodes(expr) do node
        node isa Expr || return false
        (node.head === :ref && length(node.args) == 2 && node.args[1] === name) ||
            (callee_name(node) === :getindex && length(node.args) == 3 &&
             node.args[2] === name)
    end
end

"Every occurrence of the symbol `name` in `expr`, whether it is a read or not."
occurrences_of(expr, name::Symbol) = collect_nodes(node -> node === name, expr)

function build(equation, svariables...; inplace, cse)
    _, arrays = parameter_arguments(params(snn))
    generated_expression(
        Symbolics.scalarize(equation), svariables, arrays; inplace = inplace, cse = cse)
end

@testset "out-of-place codegen, $ndata data argument(s), cse = $cse" for (ndata, equation, svariables) in [
        (1, c(snn.input, params(snn)), (snn.input,)),
        (2, (c(snn.input, params(snn)) - soutput) .^ 2, (snn.input, soutput))],
    cse in (false, true)

    expression = build(equation, svariables...; inplace = false, cse = cse)
    names, body = function_arguments_and_body(expression)

    # one argument per data variable, then one per parameter array, in that order
    @test length(names) == ndata + length(first(parameter_arguments(params(snn))))
    @test allunique(names)

    # the array constructor still takes a type as its first argument, filled in with the type of one
    # of the arguments — which is the one use of an argument that is not a read
    constructors = collect_nodes(node -> callee_name(node) === :create_array, body)
    @test !isempty(constructors)
    @test all(node -> callee_name(node.args[2]) === :typeof, constructors)

    # apart from that, every data argument is read one entry at a time and never used as a whole
    fixed = use_generic_array_constructor(body)
    for i in 1:ndata
        @test !isempty(reads_of(fixed, names[i]))
        @test length(occurrences_of(fixed, names[i])) == length(reads_of(fixed, names[i]))
    end

    # CSE must actually have happened, otherwise this is a re-run of the `cse = false` case
    @test occursin("##cse#", string(body)) == cse
end

@testset "in-place codegen, $ndata data argument(s), $name, cse = $cse" for (ndata, name, equation, svariables) in [
        (1, "vector-valued", c(snn.input, params(snn)), (snn.input,)),
        (1, "matrix-valued", derivative(Jacobian(snn)), (snn.input,)),
        (2, "vector-valued", (c(snn.input, params(snn)) - soutput) .^ 2,
            (snn.input, soutput))],
    cse in (false, true)

    expression = build(equation, svariables...; inplace = true, cse = cse)
    @test !isnothing(expression)
    names, body = function_arguments_and_body(expression)

    # the output argument is prepended and is *not* counted in the numbering of the others
    @test length(names) == 1 + ndata + length(first(parameter_arguments(params(snn))))
    output_name = names[1]

    # the output is addressed with a single linear index, whatever the shape of the equation, which
    # is what lets `accumulate_into_output` offset the writes by the batch index
    writes = collect_nodes(body) do node
        node isa Expr && node.head === :(=) && node.args[1] isa Expr &&
            node.args[1].head === :ref && node.args[1].args[1] === output_name
    end
    @test length(writes) == length(Symbolics.scalarize(equation))
    @test all(w -> length(w.args[1].args) == 2, writes)   # linear, never cartesian

    for i in 1:ndata
        @test !isempty(reads_of(body, names[1 + i]))
    end
end

# `Symbolics.build_function` emits no in-place form for a scalar equation, which is why those take
# the out-of-place path. If that ever changes, the scalar special cases become dead code.
@testset "scalar equations still have no in-place form, cse = $cse" for cse in (false, true)
    @test isnothing(build(sum(c(snn.input, params(snn))), snn.input; inplace = true, cse = cse))
end

# End-to-end guard: even if every property above holds, the assembled function must still evaluate
# correctly over a batch.
@testset "the assembled function is correct over a batch, cse = $cse" for cse in (false, true)
    f = build_nn_function(c(snn.input, params(snn)), snn; cse = cse)
    x = rand(2, 5)
    @test f(x, params(nn)) ≈ reduce(hcat, [c(x[:, k], params(nn)) for k in axes(x, 2)])
end
