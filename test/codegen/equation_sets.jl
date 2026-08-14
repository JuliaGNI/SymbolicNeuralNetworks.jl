# `build_nn_function` generates all entries of a `NamedTuple`- or `NeuralNetworkParameters`-valued
# equation set as ONE function and splits the flat result afterwards (`flatten_equations` /
# `unflatten`), instead of one function per entry: the entries of a symbolic gradient share the
# whole forward pass, which would otherwise be re-derived once per parameter array.
#
# The split has to reproduce, entry by entry, exactly what building each entry on its own produces —
# values, shapes and concrete types — for both `reduce` modes and for every input rank. Scalar
# entries are included: they used to make the whole set fall back to a separate per-entry code path.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: flatten_equations, unflatten, FlatSlice, symbolic_parameter_gradient
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, NeuralNetworkParameters
using Symbolics
using Test
import Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
soutput = Symbolics.variables(:y, 1:2)

"Rebuild a nested set from the results of applying `f` to each of its entries."
rewrap(eqs::NamedTuple, values) = NamedTuple{keys(eqs)}(values)
rewrap(eqs::NeuralNetworkParameters, values) = NeuralNetworkParameters{keys(eqs)}(values)

"Build every entry of `eqs` as its own function — the reference the joint path has to reproduce."
per_entry(eqs::Union{NamedTuple, NeuralNetworkParameters}, args...; kwargs...) =
    rewrap(eqs, Tuple(per_entry(eq, args...; kwargs...) for eq in values(eqs)))
per_entry(eq, args...; kwargs...) = build_nn_function(eq, args...; kwargs...)

evaluate_entries(fs::Union{NamedTuple, NeuralNetworkParameters}, args...) =
    rewrap(fs, Tuple(evaluate_entries(f, args...) for f in values(fs)))
evaluate_entries(f, args...) = f(args...)

function compare_entries(joint, reference)
    @test keys(joint) == keys(reference)
    for key in keys(joint)
        if joint[key] isa Union{NamedTuple, NeuralNetworkParameters}
            compare_entries(joint[key], reference[key])
        else
            @test joint[key] ≈ reference[key]
            @test size(joint[key]) == size(reference[key])
            @test typeof(joint[key]) == typeof(reference[key])
        end
    end
end

@testset "flatten_equations / unflatten round-trip" begin
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
    flat, template = flatten_equations(eqs)
    @test length(flat) == 4          # two entries of length 2
    @test template.a isa FlatSlice
    @test template.a.range == 1:2
    @test template.b.range == 3:4
    # the ranges must tile the flat vector exactly, without gaps or overlap
    @test vcat(collect(template.a.range), collect(template.b.range)) == 1:length(flat)

    # a scalar entry occupies a single slot and is recorded with an empty size
    _, scalar_template = flatten_equations((a = c(snn.input, params(snn)), s = sum(c(snn.input, params(snn)))))
    @test scalar_template.s.size == ()
    @test length(scalar_template.s.range) == 1
end

@testset "unflatten reproduces the layout of each batch shape" begin
    template = (vector = FlatSlice(1:2, (2,)), matrix = FlatSlice(3:8, (2, 3)), scalar = FlatSlice(9:9, ()))

    single = collect(1.0:9.0)                       # a single sample, or reduce = +
    @test unflatten(template, single).vector == [1.0, 2.0]
    @test unflatten(template, single).matrix == reshape(3.0:8.0, 2, 3)
    @test unflatten(template, single).scalar === 9.0

    batched = repeat(collect(1.0:9.0), 1, 4)        # reduce = hcat over four samples
    @test size(unflatten(template, batched).vector) == (2, 4)
    @test size(unflatten(template, batched).matrix) == (2, 12)
    @test size(unflatten(template, batched).scalar) == (1, 4)

    two_dimensional = repeat(collect(1.0:9.0), 1, 2, 3)
    vector_only = (vector = template.vector, scalar = template.scalar)
    @test size(unflatten(vector_only, two_dimensional).vector) == (2, 2, 3)
    @test size(unflatten(vector_only, two_dimensional).scalar) == (1, 2, 3)
    # a matrix-valued entry has no room for a second batch dimension, exactly as when it is built on
    # its own — the joint path used to return an (m·n, N₁, N₂) array instead
    @test_throws ArgumentError unflatten(template, two_dimensional)
end

@testset "NamedTuple-valued equations agree with per-entry codegen, reduce = $reduction" for reduction in (hcat, +)
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)) .^ 2)
    joint = build_nn_function(eqs, params(snn), snn.input; reduce = reduction)
    reference = per_entry(eqs, params(snn), snn.input; reduce = reduction)

    for input in (rand(3, 6), rand(3, 1), rand(3))
        compare_entries(joint(input, ps), evaluate_entries(reference, input, ps))
    end
end

@testset "two-input equations agree with per-entry codegen, reduce = $reduction" for reduction in (hcat, +)
    eqs = (a = (c(snn.input, params(snn)) - soutput) .^ 2, b = c(snn.input, params(snn)))
    joint = build_nn_function(eqs, params(snn), snn.input, soutput; reduce = reduction)
    reference = per_entry(eqs, params(snn), snn.input, soutput; reduce = reduction)

    for (input, output) in ((rand(3, 6), rand(2, 6)), (rand(3, 1), rand(2, 1)), (rand(3), rand(2)))
        compare_entries(joint(input, output, ps), evaluate_entries(reference, input, output, ps))
    end
end

@testset "nested NeuralNetworkParameters (the shape a gradient has), reduce = $reduction" for reduction in (hcat, +)
    eqs = symbolic_parameter_gradient(c(snn.input, params(snn)), snn)[1]
    @test eqs isa NeuralNetworkParameters       # two levels of nesting: layer, then W/b
    joint = build_nn_function(eqs, params(snn), snn.input; reduce = reduction)
    reference = per_entry(eqs, params(snn), snn.input; reduce = reduction)

    input = rand(3, 6)
    compare_entries(joint(input, ps), evaluate_entries(reference, input, ps))
end

# The entries are copied out of the flat result rather than viewed into it.
@testset "entries do not alias one another" begin
    eqs = (a = c(snn.input, params(snn)), b = c(snn.input, params(snn)))
    result = build_nn_function(eqs, params(snn), snn.input)(rand(3, 4), ps)
    before = copy(result.b)
    result.a .= 0
    @test result.b == before
end

# Scalar entries are folded into the joint path; they used to force the whole set onto a separate
# per-entry code path, which meant losing the shared forward pass.
@testset "equation sets containing a scalar entry, reduce = $reduction" for reduction in (hcat, +)
    eqs = (a = c(snn.input, params(snn)), s = sum(c(snn.input, params(snn))))
    joint = build_nn_function(eqs, params(snn), snn.input; reduce = reduction)
    reference = per_entry(eqs, params(snn), snn.input; reduce = reduction)

    for input in (rand(3, 4), rand(3))
        compare_entries(joint(input, ps), evaluate_entries(reference, input, ps))
    end
end

@testset "arrays of equation sets" begin
    eqs = [(a = c(snn.input, params(snn)),), (b = c(snn.input, params(snn)) .^ 3,)]
    f = build_nn_function(eqs, params(snn), snn.input)
    input = rand(3)
    result = f(input, ps)
    @test length(result) == 2
    @test result[1].a ≈ c(input, ps)
    @test result[2].b ≈ c(input, ps) .^ 3
end
