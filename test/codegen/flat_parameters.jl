# A generated function whose parameter argument is one flat vector, and the derivative with respect to
# that vector. The numeric conversions are `NeuralNetworkParameters`' — what is tested here is that
# this package's two additions agree with the structured path they wrap, and that the flat form is
# usable for the thing it exists for: differentiating with respect to a vector.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: flatten_gradient, symbolic_parameter_gradient,
                              FlatParameterFunction
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, params, FeedForwardLoss
using NeuralNetworkParameters: NetworkParameters, FlatParameters, flatten, unflatten,
                               parameterlayout, flatlength
using Symbolics
using Test
import ForwardDiff, Random

Random.seed!(123)

c = Chain(Dense(3, 4, tanh), Dense(4, 2, tanh))
nn = NeuralNetwork(c, Float64)
snn = SymbolicNeuralNetwork(c)
ps = params(nn)
w, layout = flatten(ps)
soutput = Symbolics.variables(:y, 1:2)

@testset "a flat parameter argument gives what the structured one gives" begin
    eq = c(snn.input, params(snn))
    flat = build_flat_function(eq, snn)
    structured = build_nn_function(eq, snn)

    for input in (rand(3), rand(3, 5), rand(3, 2, 2))
        @test flat(input, w) ≈ structured(input, ps)
        @test flat(input, FlatParameters(ps)) ≈ structured(input, ps)
    end
end

@testset "two data arguments" begin
    eq = (c(snn.input, params(snn)) - soutput) .^ 2
    flat = build_flat_function(eq, snn, soutput)
    structured = build_nn_function(eq, snn, soutput)

    for (input, output) in ((rand(3), rand(2)), (rand(3, 4), rand(2, 4)))
        @test flat(input, output, w) ≈ structured(input, output, ps)
    end
end

@testset "an equation set with a flat parameter argument" begin
    eqs = (a = c(snn.input, params(snn)), b = sum(c(snn.input, params(snn))))
    flat = build_flat_function(eqs, snn)
    structured = build_nn_function(eqs, snn)

    input = rand(3, 4)
    @test flat(input, w).a ≈ structured(input, ps).a
    @test flat(input, w).b ≈ structured(input, ps).b
end

# The reason for the flat form: `w` may have a different element type from the parameters the layout
# was built from, so a solver can differentiate through it.
@testset "the flat form differentiates" begin
    loss = FeedForwardLoss()
    input, output = rand(3), rand(2)
    f = build_flat_function(loss(snn.model, params(snn), snn.input, soutput), snn, soutput)

    @test f(input, output, w) ≈ loss(c, ps, input, output)
    gradient = ForwardDiff.gradient(v -> f(input, output, v), w)
    reference = build_nn_function(
        flat_parameter_gradient(loss(snn.model, params(snn), snn.input,
                soutput), snn),
        params(snn), snn.input, soutput)(input, output, ps)
    @test gradient ≈ reference
end

@testset "flat_parameter_gradient of a scalar is the flattened gradient" begin
    scalar = sum(c(snn.input, params(snn)))
    flat = flat_parameter_gradient(scalar, snn)
    nested, _ = flatten(symbolic_parameter_gradient(scalar, snn))

    @test flat isa Vector{Num}
    @test length(flat) == flatlength(ps)
    @test all(isequal.(flat, nested))
end

@testset "flat_parameter_gradient of an array is the Jacobian with respect to the flat parameters" begin
    eq = c(snn.input, params(snn))
    J = flat_parameter_gradient(eq, snn)
    @test size(J) == (2, flatlength(ps))

    built = build_nn_function(J, snn)
    input = rand(3)
    # row i is the gradient of entry i, which `ForwardDiff` can produce independently
    for i in 1:2
        reference = ForwardDiff.gradient(v -> build_flat_function(eq, snn)(input, v)[i], w)
        @test built(input, ps)[i, :] ≈ reference
    end
end

# A column block of the Jacobian belongs to one entry of the parameter set, and `unflatten`'s matrix
# method is what reads it back — the one place this package leaves that meaning to upstream.
@testset "the Jacobian splits back into parameter blocks" begin
    J = build_nn_function(flat_parameter_gradient(c(snn.input, params(snn)), snn), snn)(rand(3), ps)
    blocks = unflatten(layout, permutedims(J))
    @test size(blocks.L1.W) == (length(ps.L1.W), 2)
    @test size(blocks.L2.b) == (length(ps.L2.b), 2)
end

# Neither function reads a model, so degrees of freedom that are not a network's parameters go
# through both. This is what `docs/src/guide/flat_parameters.md` claims at the end, exercised.
@testset "degrees of freedom that are not a network's" begin
    dof = NetworkParameters((scale = Symbolics.variables(:s, 1:2),
        offset = Symbolics.variables(:o, 1:2, 1:2)))
    sinput = Symbolics.variables(:t, 1:2)
    # a nonlinear expression over `dof` that no `Chain` produces
    equation = dof.offset * (dof.scale .* sinput) .- sin.(dof.scale)

    residual = build_flat_function(equation, dof, sinput)
    J = flat_parameter_gradient(equation, dof)
    @test size(J) == (2, flatlength(dof))
    jacobian = build_flat_function(J, dof, sinput)

    numbers = NetworkParameters((scale = [0.5, 2.0], offset = [1.0 2.0; 3.0 4.0]))
    v, dof_layout = flatten(numbers)
    t = rand(2)

    reference(u) =
        let p = unflatten(dof_layout, u)
            p.offset * (p.scale .* t) .- sin.(p.scale)
        end
    @test residual(t, v) ≈ reference(v)
    @test jacobian(t, v) ≈ ForwardDiff.jacobian(reference, v)
end

# `SymbolicNeuralNetwork`'s parameter field is a `Union{NetworkParameters, NamedTuple}` and
# `symbolic_differentials` walks either, so the gradient functions have to accept a bare `NamedTuple`
# too — both as the parameters of a network and as degrees of freedom handed over directly.
@testset "parameters nested in a plain NamedTuple" begin
    nt = NamedTuple(params(snn))
    snt = SymbolicNeuralNetwork(snn.architecture, snn.model, nt, snn.input)
    @test params(snt) isa NamedTuple

    eq = c(snt.input, params(snt))
    J = flat_parameter_gradient(eq, snt)
    @test size(J) == (2, flatlength(ps))
    # the same variables either way round, so the two derivatives are the same expressions
    @test all(isequal.(J, flat_parameter_gradient(c(snn.input, params(snn)), snn)))
    @test all(isequal.(flat_parameter_gradient(eq, nt), J))
    @test all(isequal.(flatten_gradient(symbolic_parameter_gradient(eq, nt)), J))
end

@testset "flatten_gradient rejects an empty expression" begin
    @test_throws ArgumentError flatten_gradient(NetworkParameters[])
end

@testset "show" begin
    f = build_flat_function(c(snn.input, params(snn)), snn)
    @test f isa FlatParameterFunction{1}
    @test occursin("$(flatlength(ps)) parameters", repr(f))
end
