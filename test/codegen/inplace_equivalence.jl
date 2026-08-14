# The in-place kernel addresses its output linearly and offsets the writes by the batch index, so the
# layout of the result is reproduced by arithmetic rather than by `hcat`/`+`. These tests pin that
# layout against the out-of-place path, which builds it with `Base.reduce` — same values *and* same
# shape — for every combination of equation shape, input shape and `reduce` mode.
#
# Some combinations are not supported by either path (a three-dimensional input needs a batch
# dimension to put the results in, which a matrix-valued equation has already used); those have to
# fail the same way, which is what `compare_or_both_fail` checks.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, Gradient, derivative
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using Symbolics
using Test
import Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
snn = SymbolicNeuralNetwork(c)
nn = NeuralNetwork(c)
ps = params(nn)
soutput = Symbolics.variables(:y, 1:2)

"""
Compare the in-place and out-of-place results, requiring both value *and* shape to agree. If the
out-of-place path does not support the combination either, require that the in-place path fails too
rather than silently returning something different.
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

@testset "single input, $name, reduce = $reduction" for
        (name, eq) in ("vector-valued" => c(snn.input, params(snn)),
                       "matrix-valued (Jacobian)" => derivative(Jacobian(snn)),
                       "matrix-valued (Gradient)" => derivative(Gradient(snn))[1].L1.W),
        reduction in (hcat, +)

    f_iip = build_nn_function(eq, params(snn), snn.input; reduce = reduction)
    f_oop = build_nn_function(eq, params(snn), snn.input; reduce = reduction, inplace = false)

    compare_or_both_fail(f_iip, f_oop, rand(3, 6), ps)       # batch
    compare_or_both_fail(f_iip, f_oop, rand(3, 1), ps)       # batch of one
    compare_or_both_fail(f_iip, f_oop, rand(3), ps)          # single vector
    compare_or_both_fail(f_iip, f_oop, rand(3, 2, 3), ps)    # two batch dimensions
end

@testset "two inputs, $name, reduce = $reduction" for
        (name, eq) in ("vector-valued" => (c(snn.input, params(snn)) - soutput) .^ 2,
                       "matrix-valued" => derivative(Gradient((c(snn.input, params(snn)) - soutput) .^ 2, snn))[1].L1.W),
        reduction in (hcat, +)

    f_iip = build_nn_function(eq, params(snn), snn.input, soutput; reduce = reduction)
    f_oop = build_nn_function(eq, params(snn), snn.input, soutput; reduce = reduction, inplace = false)

    compare_or_both_fail(f_iip, f_oop, rand(3, 6), rand(2, 6), ps)
    compare_or_both_fail(f_iip, f_oop, rand(3, 1), rand(2, 1), ps)
    compare_or_both_fail(f_iip, f_oop, rand(3), rand(2), ps)
    compare_or_both_fail(f_iip, f_oop, rand(3, 2, 3), rand(2, 2, 3), ps)
end
