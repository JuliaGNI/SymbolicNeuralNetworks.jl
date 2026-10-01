using JET
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: Jacobian, derivative, symbolic_parameter_gradient,
                              promoted_eltype, split_result
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params
using NeuralNetworkParameters: NetworkParameters
using Test

# Static optimisation analysis of the hot paths: every function of `src/` that
# `test/codegen/allocations.jl` asserts with `@allocated`, at the concrete argument types that file
# passes. Those are `promoted_eltype` (through `eltype_folds`), the call of an
# `InPlaceBatchedFunction`, the call of an `EquationSetFunction` and `split_result`, each on a
# single sample and on a batch. Each further element type that a test outside `test/quality/`
# passes directly to the same method gets one line too. An element type that reaches it only
# through another function gets none: the `ForwardDiff.Dual` parameters of
# `test/codegen/zygote_differentiability.jl` reach the call of an `InPlaceBatchedFunction` through
# `ForwardDiff.gradient`.

if isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
    m = (SymbolicNeuralNetworks,)

    # test/codegen/allocations.jl: the `ShallowNet` basis, a sample and a batch of `Float64`
    c = Chain(Dense(1, 4, tanh), Dense(4, 1, identity; use_bias = false))
    snn = SymbolicNeuralNetwork(c)
    P = typeof(params(NeuralNetwork(c)))
    f = build_nn_function(derivative(Jacobian(snn)), snn)
    g = build_nn_function(symbolic_parameter_gradient(c(snn.input, params(snn))[1], snn), snn)
    single = typeof(g.f([0.5], params(NeuralNetwork(c))))
    batched = typeof(g.f(ones(1, 8), params(NeuralNetwork(c))))

    @test isempty(JET.get_reports(JET.report_opt(promoted_eltype, (Vector{Float64}, P);
        target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(promoted_eltype, (Matrix{Float64}, P);
        target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(f, (Vector{Float64}, P); target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(f, (Matrix{Float64}, P); target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(g, (Vector{Float64}, P); target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(g, (Matrix{Float64}, P); target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(split_result, (typeof(g.layout), single);
        target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(split_result, (typeof(g.layout), batched);
        target_modules = m)))

    # test/codegen/batched_function.jl: the network of that file
    cb = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
    snnb = SymbolicNeuralNetwork(cb)
    fb = build_nn_function(cb(snnb.input, params(snnb)), snnb)
    Pb32 = typeof(params(NeuralNetwork(cb, Float32)))
    Pint = typeof(NetworkParameters((L1 = (W = ones(Int, 4, 3), b = zeros(Int, 4)),
        L2 = (W = ones(Int, 2, 4), b = zeros(Int, 2)))))

    # test/codegen/batched_function.jl:181: `Float32` input and parameters
    @test isempty(JET.get_reports(JET.report_opt(promoted_eltype, (Vector{Float32}, Pb32);
        target_modules = m)))
    # test/codegen/batched_function.jl:182: an `Int` named tuple and a `Float64` vector
    @test isempty(JET.get_reports(JET.report_opt(promoted_eltype,
        (NamedTuple{(:a,), Tuple{Vector{Int}}}, Vector{Float64}); target_modules = m)))

    # test/codegen/batched_function.jl:161 and :174: a `Float32` batch and an `Int` batch
    @test isempty(JET.get_reports(JET.report_opt(fb, (Matrix{Float32}, Pb32); target_modules = m)))
    @test isempty(JET.get_reports(JET.report_opt(fb, (Matrix{Int}, Pint); target_modules = m)))

    # test/derivatives/jacobian.jl:21: the Jacobian of a `Float32` network on a `Float32` sample
    cj = Chain(Dense(2, 1, tanh))
    snnj = SymbolicNeuralNetwork(cj)
    fj = build_nn_function(derivative(Jacobian(snnj)), snnj)
    Pj32 = typeof(params(NeuralNetwork(cj, Float32)))
    @test isempty(JET.get_reports(JET.report_opt(fj, (Vector{Float32}, Pj32); target_modules = m)))
else
    @test_skip "JET does not work on this Julia version"  # aviatesk/JET.jl#681
end
