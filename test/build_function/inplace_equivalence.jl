# `build_nn_function` evaluates a batch with an *in-place* kernel that writes every column
# into one preallocated array (see `_build_nn_function_iip`). It used to call an out-of-place
# kernel once per column and combine the results with `Base.reduce(reduce, …)`.
#
# The in-place kernel addresses its output linearly and offsets the writes by the batch index,
# so the layout of the result is reproduced by arithmetic rather than by `hcat`/`+`. These
# tests pin that layout down against the out-of-place path it replaced — same values *and* same
# shape — for every combination of equation shape, input shape and `reduce` mode.
#
# Some combinations are not supported by either path (a 3-D input is reshaped assuming a
# vector-valued equation); those are expected to fail the same way, which is what
# `compare_or_both_fail` checks.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative, _build_nn_function,
                              _oop_batch_wrapper, _oop_batch_wrapper2
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Symbolics: @variables
using Test
import Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)

"""
Compare the in-place and out-of-place results, requiring both value *and* shape to agree.
If the out-of-place path does not support the combination either, require that the in-place
path fails too rather than silently returning something different.
"""
function compare_or_both_fail(f_iip, f_oop, args...)
    oop_result = try
        f_oop(args...)
    catch
        @test_throws Exception f_iip(args...)
        return
    end
    iip_result = f_iip(args...)
    @test iip_result ≈ oop_result
    @test size(iip_result) == size(oop_result)
end

@testset "single input, $name, reduce = $reduce" for
        (name, eq) in ("vector-valued" => c(snn.input, params(snn)),
                       "matrix-valued (Jacobian)" => derivative(Jacobian(snn)),
                       "matrix-valued (Gradient)" => derivative(Gradient(snn))[1].L1.W),
        reduce in (hcat, +)

    f_iip = build_nn_function(eq, params(snn), snn.input; reduce = reduce)
    f_oop = _oop_batch_wrapper(_build_nn_function(eq, params(snn), snn.input), reduce)

    compare_or_both_fail(f_iip, f_oop, rand(3, 6), ps)       # batch
    compare_or_both_fail(f_iip, f_oop, rand(3, 1), ps)       # batch of one
    compare_or_both_fail(f_iip, f_oop, rand(3), ps)          # single vector
    compare_or_both_fail(f_iip, f_oop, rand(3, 2, 3), ps)    # 3-D input
end

@variables soutput[1:2]

@testset "two inputs, $name, reduce = $reduce" for
        (name, eq) in ("vector-valued" => (c(snn.input, params(snn)) - soutput) .^ 2,
                       "matrix-valued" => derivative(Gradient((c(snn.input, params(snn)) - soutput) .^ 2, snn))[1].L1.W),
        reduce in (hcat, +)

    f_iip = build_nn_function(eq, params(snn), snn.input, soutput; reduce = reduce)
    f_oop = _oop_batch_wrapper2(_build_nn_function(eq, params(snn), snn.input, soutput), reduce)

    compare_or_both_fail(f_iip, f_oop, rand(3, 6), rand(2, 6), ps)
    compare_or_both_fail(f_iip, f_oop, rand(3, 1), rand(2, 1), ps)
    compare_or_both_fail(f_iip, f_oop, rand(3), rand(2), ps)
    compare_or_both_fail(f_iip, f_oop, rand(3, 2, 3), rand(2, 2, 3), ps)
end

# The output array is allocated before the kernel runs, so its element type comes from
# `promoted_eltype` rather than from the result. `Float32` parameters must not be silently
# widened to `Float64` (nor the other way round).
@testset "element type follows the inputs" begin
    f = build_nn_function(c(snn.input, params(snn)), params(snn), snn.input)
    nn32 = NeuralNetwork(c, Float32)
    @test eltype(f(rand(Float32, 3, 4), params(nn32))) == Float32
    @test eltype(f(rand(3, 4), ps)) == Float64
    # mixed inputs promote
    @test eltype(f(rand(Float32, 3, 4), ps)) == Float64
end
