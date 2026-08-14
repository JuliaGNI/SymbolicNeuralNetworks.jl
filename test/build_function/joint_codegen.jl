# `build_nn_function` generates all entries of a `NamedTuple`- or
# `NeuralNetworkParameters`-valued equation set as ONE function and splits the flat result
# afterwards (`flatten_eqs` / `unflatten`), instead of one function per entry.
#
# The split has to reproduce, entry by entry, exactly what the per-entry path produced —
# values, shapes and concrete types — for both `reduce` modes and for batched as well as
# single-vector inputs. `_build_nn_function_per_leaf` is still around as the fallback for
# scalar-valued entries, which makes it the reference to compare against.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: _build_nn_function_per_leaf, flatten_eqs, unflatten, FlatSlice
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, NeuralNetworkParameters
using Symbolics
using Symbolics: @variables
using Test
import Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)

function compare_entries(joint, per_leaf)
    @test keys(joint) == keys(per_leaf)
    for key in keys(joint)
        if joint[key] isa Union{NamedTuple, NeuralNetworkParameters}
            compare_entries(joint[key], per_leaf[key])
        else
            @test joint[key] ≈ per_leaf[key]
            @test size(joint[key]) == size(per_leaf[key])
            @test typeof(joint[key]) == typeof(per_leaf[key])
        end
    end
end

@testset "flatten_eqs / unflatten round-trip" begin
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
    flat, template = flatten_eqs(eqs)
    @test length(flat) == 4          # two entries of length 2
    @test template.a isa FlatSlice
    @test template.a.range == 1:2
    @test template.b.range == 3:4
    # the ranges must tile the flat vector exactly, without gaps or overlap
    @test vcat(collect(template.a.range), collect(template.b.range)) == 1:length(flat)

    # scalar entries have no in-place kernel, so the set must be rejected
    @variables scalar_eq
    @test isnothing(flatten_eqs((a = c(snn.input, params(snn)), b = scalar_eq)))
end

@testset "NamedTuple-valued equations agree with per-entry codegen, reduce = $reduce" for reduce in (hcat, +)
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
    joint = build_nn_function(eqs, params(snn), snn.input; reduce = reduce)
    per_leaf = _build_nn_function_per_leaf(eqs, params(snn), snn.input; reduce = reduce)

    for input in (rand(3, 6), rand(3, 1), rand(3))
        compare_entries(joint(input, ps), per_leaf(input, ps))
    end
end

@variables soutput[1:2]

@testset "two-input equations agree with per-entry codegen, reduce = $reduce" for reduce in (hcat, +)
    eqs = (a = (c(snn.input, params(snn)) - soutput) .^ 2, b = c(snn.input, params(snn)))
    joint = build_nn_function(eqs, params(snn), snn.input, soutput; reduce = reduce)
    per_leaf = _build_nn_function_per_leaf(eqs, params(snn), snn.input, soutput; reduce = reduce)

    for (input, output) in ((rand(3, 6), rand(2, 6)), (rand(3, 1), rand(2, 1)), (rand(3), rand(2)))
        compare_entries(joint(input, output, ps), per_leaf(input, output, ps))
    end
end

@testset "nested NeuralNetworkParameters (the shape a pullback has), reduce = $reduce" for reduce in (hcat, +)
    eqs = SymbolicNeuralNetworks.symbolic_pullback(c(snn.input, params(snn)), snn)[1]
    @test eqs isa NeuralNetworkParameters       # two levels of nesting: layer, then W/b
    joint = build_nn_function(eqs, params(snn), snn.input; reduce = reduce)
    per_leaf = _build_nn_function_per_leaf(eqs, params(snn), snn.input; reduce = reduce)

    input = rand(3, 6)
    compare_entries(joint(input, ps), per_leaf(input, ps))
end

@testset "entries do not alias one another" begin
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)))
    result = build_nn_function(eqs, params(snn), snn.input)(rand(3, 4), ps)
    before = copy(result.b)
    result.a .= 0
    @test result.b == before
end

@testset "equation sets containing a scalar fall back to per-entry codegen" begin
    # `collect` before reducing: reductions over an un-scalarized `Symbolics.Arr` produce an
    # `arrayop`, whose generated code refers to variables that nothing binds (independent of
    # `cse`) — see `build_function_generated`.
    scalar_eq = sum(collect(c(snn.input, params(snn))))
    eqs = (a = c(snn.input, params(snn)), b = scalar_eq)
    @test isnothing(flatten_eqs(eqs))    # the joint path must decline this set

    f = build_nn_function(eqs, params(snn), snn.input)
    input = rand(3, 4)
    @test f(input, ps).a ≈ reduce(hcat, [c(input[:, k], ps) for k in axes(input, 2)])
    @test vec(f(input, ps).b) ≈ [sum(c(input[:, k], ps)) for k in axes(input, 2)]
end
