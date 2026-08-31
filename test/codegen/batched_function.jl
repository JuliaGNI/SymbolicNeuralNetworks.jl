# The shapes an `AbstractBatchedFunction` produces, for every combination of
#
#   {scalar, vector, matrix equation} × {one, two data arguments} × {vector, matrix, 3-tensor input}
#   × {hcat, +} × {in-place, out-of-place},
#
# checked against an explicit per-sample reference rather than against the other implementation, so
# that the two paths cannot agree on something wrong.
#
# Two of these combinations used to be broken: a three-dimensional input with `reduce = +` threw a
# `DimensionMismatch` in the one-argument case (the two-argument one handled it), and the
# two-argument methods required both arguments to have the *same* type, so a `Vector` input with a
# `SubArray` target failed.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative, promoted_eltype, allocate_batch_output,
                              allocate_single_output, InPlaceBatchedFunction,
                              OutOfPlaceBatchedFunction
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters
using Symbolics
using Test
import Random

Random.seed!(123)

const INPUT_DIM = 3
const OUTPUT_DIM = 2

c = Chain(Dense(INPUT_DIM, 4, tanh), Dense(4, OUTPUT_DIM, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
soutput = Symbolics.variables(:y, 1:OUTPUT_DIM)

# equations of every rank, together with the function that evaluates one sample of them numerically
const EQUATIONS = [
    ("vector-valued", c(snn.input, params(snn)), (x, p) -> c(x, p)),
    ("matrix-valued", derivative(Jacobian(snn)), (x, p) -> _jacobian(x, p)),
    ("scalar-valued", sum(c(snn.input, params(snn))), (x, p) -> sum(c(x, p)))
]

# finite-difference-free reference for the Jacobian of a two-layer tanh network
function _jacobian(x, p)
    h = tanh.(p.L1.W * x .+ p.L1.b)
    o = tanh.(p.L2.W * h .+ p.L2.b)
    ((1 .- o .^ 2) .* p.L2.W) * ((1 .- h .^ 2) .* p.L1.W)
end

"The result the batched function has to reproduce, assembled from per-sample results."
function reference(evaluate, ps, input::AbstractMatrix, reduction)
    samples = [evaluate(input[:, k], ps) for k in axes(input, 2)]
    Base.reduce(reduction, samples)
end

@testset "$name, one data argument, reduce = $reduction, inplace = $inplace" for (name, eq, evaluate) in EQUATIONS,
    reduction in (hcat, +), inplace in (true, false)

    f = build_nn_function(eq, snn; reduce = reduction, inplace = inplace)
    input = rand(INPUT_DIM, 5)

    @test f(input, ps) ≈ reference(evaluate, ps, input, reduction)
    @test f(input[:, 1:1], ps) ≈ reference(evaluate, ps, input[:, 1:1], reduction)
    # a single sample keeps the shape of the equation, whatever the reduction
    @test f(input[:, 1], ps) ≈ evaluate(input[:, 1], ps)
end

@testset "two data arguments, reduce = $reduction, inplace = $inplace" for reduction in (hcat, +),
    inplace in (true, false)

    eq = (c(snn.input, params(snn)) - soutput) .^ 2
    evaluate(x, y, p) = (c(x, p) - y) .^ 2
    f = build_nn_function(eq, snn, soutput; reduce = reduction, inplace = inplace)

    input, output = rand(INPUT_DIM, 5), rand(OUTPUT_DIM, 5)
    @test f(input, output, ps) ≈
          Base.reduce(reduction, [evaluate(input[:, k], output[:, k], ps)
                                  for k in axes(input, 2)])
    @test f(input[:, 1], output[:, 1], ps) ≈ evaluate(input[:, 1], output[:, 1], ps)

    # the two data arguments need not be of the same type
    @test f(input[:, 1], view(output, :, 1), ps) ≈ evaluate(input[:, 1], output[:, 1], ps)
    @test f(view(input, :, 1:3), output[:, 1:3], ps) ≈ f(input[:, 1:3], output[:, 1:3], ps)
end

# A scalar-valued equation used to throw `Cannot call tail on an empty tuple` here, because its
# size is `()` and the reshape asked for the product of the tail of that.
@testset "three-dimensional input, $name, reduce = $reduction, inplace = $inplace" for (name, eq, evaluate) in EQUATIONS,
    reduction in (hcat, +), inplace in (true, false)

    f = build_nn_function(eq, snn; reduce = reduction, inplace = inplace)
    input = rand(INPUT_DIM, 2, 3)
    flat = reshape(input, INPUT_DIM, 6)

    if reduction === +
        # the samples were summed, so there is no batch dimension left to restore
        @test f(input, ps) ≈ reference(evaluate, ps, flat, +)
    elseif name == "matrix-valued"
        # concatenating a matrix-valued result already uses the second dimension
        @test_throws ArgumentError f(input, ps)
    else
        result = f(input, ps)
        # a scalar-valued equation counts as one of size m = 1
        @test size(result) == (name == "scalar-valued" ? 1 : OUTPUT_DIM, 2, 3)
        for i in 1:2, j in 1:3

            @test vec(result[:, i, j]) ≈ [evaluate(input[:, i, j], ps);]
        end
    end
end

@testset "a matrix-valued equation cannot use two batch dimensions" begin
    f = build_nn_function(derivative(Jacobian(snn)), snn)
    @test_throws ArgumentError f(rand(INPUT_DIM, 2, 3), ps)
    # ... but summing over the batch is fine, as that leaves no batch dimension
    g = build_nn_function(derivative(Jacobian(snn)), snn; reduce = +)
    @test g(rand(INPUT_DIM, 2, 3), ps) isa Matrix

    # the equation-set path applies the same restriction, entry by entry
    h = build_nn_function((j = derivative(Jacobian(snn)),), params(snn), snn.input)
    @test_throws ArgumentError h(rand(INPUT_DIM, 2, 3), ps)
end

# `Base.reduce` has nothing to fold over an empty batch, so the out-of-place path used to throw
# where the in-place one returned an empty result.
@testset "an empty batch, $name, reduce = $reduction" for (name, eq, evaluate) in EQUATIONS,
    reduction in (hcat, +)

    f_iip = build_nn_function(eq, snn; reduce = reduction)
    f_oop = build_nn_function(eq, snn; reduce = reduction, inplace = false)
    empty_input = rand(INPUT_DIM, 0)

    @test f_iip(empty_input, ps) == f_oop(empty_input, ps)
    if reduction === hcat
        @test size(f_oop(empty_input, ps), 2) == 0
    else
        # nothing was summed, so the result is the zero of the shape of the equation
        @test all(iszero, f_oop(empty_input, ps))
    end
end

# Anything else than all-vectors, all-matrices or all-3-tensors used to surface as a `MethodError`
# naming an internal function and the whole `RuntimeGeneratedFunction` type.
@testset "data arguments of other or mixed ranks are rejected" begin
    f = build_nn_function(c(snn.input, params(snn)), snn)
    @test_throws ArgumentError f(rand(INPUT_DIM, 2, 2, 2), ps)

    g = build_nn_function((c(snn.input, params(snn)) - soutput) .^ 2, snn, soutput)
    @test_throws ArgumentError g(rand(INPUT_DIM), rand(OUTPUT_DIM, 4), ps)
end

@testset "mismatched batch sizes are rejected" begin
    f = build_nn_function((c(snn.input, params(snn)) - soutput) .^ 2, snn, soutput)
    @test_throws DimensionMismatch f(rand(INPUT_DIM, 3), rand(OUTPUT_DIM, 4), ps)
end

# The result of the in-place path is allocated before the kernel runs, so its element type comes
# from `promoted_eltype` rather than from the result.
@testset "element type follows the inputs" begin
    f = build_nn_function(c(snn.input, params(snn)), snn)
    nn32 = NeuralNetwork(c, Float32)
    @test eltype(f(rand(Float32, INPUT_DIM, 4), params(nn32))) == Float32
    @test eltype(f(rand(INPUT_DIM, 4), ps)) == Float64
    @test eltype(f(rand(Float32, INPUT_DIM, 4), ps)) == Float64
end

# An equation over integer inputs does not evaluate to an integer, so the promoted type has to be
# widened before the kernel writes into the array — otherwise this is an `InexactError`.
@testset "integer inputs are widened, reduce = $reduction" for reduction in (hcat, +)
    f_iip = build_nn_function(c(snn.input, params(snn)), snn; reduce = reduction)
    f_oop = build_nn_function(c(snn.input, params(snn)), snn; reduce = reduction, inplace = false)
    int_ps = NetworkParameters((L1 = (W = ones(Int, 4, INPUT_DIM), b = zeros(Int, 4)),
        L2 = (W = ones(Int, OUTPUT_DIM, 4), b = zeros(Int, OUTPUT_DIM))))
    int_input = [1 2; 3 4; 5 6]
    @test f_iip(int_input, int_ps) ≈ f_oop(int_input, int_ps)
    @test eltype(f_iip(int_input, int_ps)) == Float64
    @test eltype(f_iip(rand(INPUT_DIM, 2), int_ps)) == Float64
end

@testset "promoted_eltype and the allocators" begin
    @test promoted_eltype(rand(Float32, 2), 1.0) == Float64
    @test promoted_eltype(rand(Float32, 2), params(NeuralNetwork(c, Float32))) == Float32
    @test promoted_eltype((a = rand(Int, 2),), rand(2)) == Float64

    @test size(allocate_batch_output(Float64, (2,), 5, hcat)) == (2, 5)
    @test size(allocate_batch_output(Float64, (2, 3), 5, hcat)) == (2, 15)
    @test size(allocate_batch_output(Float64, (2, 3), 5, +)) == (2, 3)
    @test allocate_batch_output(Float64, (2, 3), 5, +) == zeros(2, 3)
    @test eltype(allocate_batch_output(Int, (2,), 5, hcat)) == Float64
    @test eltype(allocate_batch_output(Bool, (2,), 5, hcat)) == Float64
    @test size(allocate_single_output(Float64, (2, 3), hcat)) == (2, 3)
end

@testset "the built functions describe themselves" begin
    @test occursin("InPlaceBatchedFunction", sprint(show, build_nn_function(c(snn.input, params(snn)), snn)))
    @test occursin("OutOfPlaceBatchedFunction",
        sprint(show, build_nn_function(c(snn.input, params(snn)), snn; inplace = false)))
    @test build_nn_function(c(snn.input, params(snn)), snn) isa InPlaceBatchedFunction
    @test build_nn_function(sum(c(snn.input, params(snn))), snn) isa
          OutOfPlaceBatchedFunction
end
