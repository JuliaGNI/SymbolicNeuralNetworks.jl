# Reproduces the numbers that motivated the code-generation choices in `src/codegen/`.
#
# Three things are measured, for networks of increasing depth:
#
#   1. `Symbolics.build_function` with and without common subexpression elimination — time and
#      size of the emitted code. Symbolics stores expressions as a hash-consed DAG but prints
#      them as a tree, so without CSE every reuse of a subexpression is emitted again and the
#      code grows exponentially with depth.
#   2. `SymbolicPullback` construction, which is dominated by that code generation.
#   3. Evaluating the pullback over a batch — time and allocations.
#   4. The layerwise construction against the monolithic one — the size of the symbolic material
#      each has to hold, and what each costs to build. This is the measurement behind issue #49:
#      CSE addresses the *emitted code*, and the expression is the other half of the problem.
#
# Run with
#     julia --project=. scripts/codegen_comparison.jl
# Add `--all` to also generate the deep networks *without* CSE. Be warned: for 5-10-10-10-1 that
# emits roughly 440 MB of code and takes minutes, which is exactly the point being made.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: generated_expression, parameter_arguments, symbolic_parameter_gradient,
                              symbolic_derivative, symbolic_differentials, layer_seed
using AbstractNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, FeedForwardLoss, params, output_dimension
using NeuralNetworkParameters: NetworkParameters
using Symbolics
using Printf
import Random

Random.seed!(123)

const RUN_EVERYTHING = "--all" ∈ ARGS
const ARCHITECTURES = ((5, 10, 1), (5, 10, 10, 1), (5, 10, 10, 10, 1))
# the networks of issue #49: depth at a fixed width, then width at a fixed depth
const LAYERWISE_ARCHITECTURES = ((2, 4, 2), (2, 4, 4, 2), (2, 4, 4, 4, 2), (2, 4, 4, 4, 4, 2),
                                 (2, 4, 4, 4, 4, 4, 2), (2, 8, 8, 8, 2), (2, 16, 16, 16, 2))
# generating the pullback of a three-hidden-layer network without CSE is not practical
const MAX_LAYERS_WITHOUT_CSE = 3
const BATCH_SIZE = 100

chain(dims) = Chain((Dense(dims[i], dims[i + 1], tanh) for i in 1:(length(dims) - 1))...)

skip_without_cse(dims) = !RUN_EVERYTHING && length(dims) > MAX_LAYERS_WITHOUT_CSE

"Time `Symbolics.build_function` and measure the size of the code it emits, per parameter block."
function measure_codegen(dims, cse::Bool)
    c = chain(dims)
    snn = SymbolicNeuralNetwork(c)
    soutput = Symbolics.variables(:y, 1:dims[end])
    gradient = symbolic_parameter_gradient(FeedForwardLoss()(c, params(snn), snn.input, soutput), snn)
    _, arrays = parameter_arguments(params(snn))

    seconds = 0.0
    characters = 0
    for layer in keys(gradient), array in keys(gradient[layer])
        eq = gradient[layer][array]
        seconds += @elapsed code = generated_expression(eq, (snn.input, soutput), arrays;
                                                        inplace = false, cse = cse)
        characters += length(string(code))
    end
    (; seconds, characters)
end

"Time `SymbolicPullback` end to end, then evaluate it over a batch."
function measure_pullback(dims, cse::Bool; layerwise = :auto)
    c = chain(dims)
    snn = SymbolicNeuralNetwork(c)
    nn = NeuralNetwork(c)
    ps = params(nn)

    construction = @elapsed pb = SymbolicPullback(snn, FeedForwardLoss(); cse = cse,
                                                 layerwise = layerwise)

    input = rand(dims[1], BATCH_SIZE)
    output = rand(dims[end], BATCH_SIZE)
    evaluate() = pb(ps, c, (input, output))[2](1.0)
    evaluate()  # compile the generated function before timing it
    seconds = minimum(@elapsed(evaluate()) for _ in 1:10)
    (; construction, seconds, bytes = @allocated evaluate())
end

"""
    nodes(expression)

The number of nodes in a symbolic expression, counting a reused subexpression once per use — which is
how `Symbolics.build_function` prints it, and therefore the size that matters.
"""
nodes(x::Num) = nodes(Symbolics.value(x))
nodes(x) = Symbolics.iscall(x) ? 1 + sum(nodes, Symbolics.arguments(x); init = 0) : 1
nodes(xs::AbstractArray) = sum(nodes, xs; init = 0)
nodes(nt::NamedTuple) = sum(nodes, values(nt); init = 0)
nodes(p::NetworkParameters) = nodes(params(p))

"The size of the one expression the monolithic construction differentiates."
function monolithic_nodes(dims)
    c = chain(dims)
    snn = SymbolicNeuralNetwork(c)
    soutput = Symbolics.variables(:y, 1:dims[end])
    nodes(symbolic_parameter_gradient(FeedForwardLoss()(c, params(snn), snn.input, soutput), snn))
end

"""
The symbolic material the layerwise construction holds: for each layer, the two derivatives of the
seeded scalar `λ · f(x; θ)`. Fresh variables at each seam are what makes this a sum over layers.

The seed comes from `layer_seed`, which is what `layer_step` builds from, so this measures the
construction rather than a second copy of it.

Both derivatives are counted for every layer, including the first layer's derivative with respect to
its input — which `layer_step` does not actually generate, since the sweep never calls it. That keeps
this column the like-for-like comparison with the monolithic one, and with the table in issue #49; the
saving is one constant, not a term that grows with the network.
"""
function layerwise_nodes(dims)
    c = chain(dims)
    sparams = params(SymbolicNeuralNetwork(c))
    total = 0
    for (layer, key) in zip(AbstractNeuralNetworks.layers(c), keys(sparams))
        seed, layer_params, sdata, _ = layer_seed(layer, key, sparams[key])
        # `sdata` is the seam's data variables, the state first; a `Dense` carries nothing beside it
        total += nodes(symbolic_derivative(seed, symbolic_differentials(first(sdata))))
        total += nodes(symbolic_derivative(seed, symbolic_differentials(layer_params[key])))
    end
    total
end

"""
Whether to skip the monolithic side. Only the widest network here is genuinely out of reach — its
gradient expression has 2·10⁸ nodes — but building and counting the others is still slow enough to be
worth a flag.
"""
skip_monolithic(dims) = !RUN_EVERYTHING && maximum(dims) > 8

println("Julia $(VERSION), $(BATCH_SIZE)-column batches\n")

# compile `build_function` and the pullback machinery before anything is timed
measure_codegen(ARCHITECTURES[1], true)
measure_pullback(ARCHITECTURES[1], true; layerwise = false)
measure_pullback(ARCHITECTURES[1], true; layerwise = true)

println("Code generation for the pullback, summed over all parameter blocks")
@printf("%-20s %12s %12s %14s %14s %8s\n", "layers", "no cse (s)", "cse (s)", "no cse (chars)", "cse (chars)", "ratio")
for dims in ARCHITECTURES
    with = measure_codegen(dims, true)
    if skip_without_cse(dims)
        @printf("%-20s %12s %12.3f %14s %14d %8s\n", dims, "skipped", with.seconds, "skipped", with.characters, "-")
    else
        without = measure_codegen(dims, false)
        @printf("%-20s %12.3f %12.3f %14d %14d %8.1f\n",
            dims, without.seconds, with.seconds, without.characters, with.characters,
            without.characters / with.characters)
    end
end

println("\nSymbolicPullback: construction, then one batch")
@printf("%-20s %12s %12s %12s\n", "layers", "build (s)", "eval (ms)", "alloc (KiB)")
for dims in ARCHITECTURES
    result = measure_pullback(dims, true; layerwise = false)
    @printf("%-20s %12.2f %12.3f %12.1f\n", dims, result.construction, result.seconds * 1e3, result.bytes / 1024)
end

println("\nSymbolic material held by each construction, in expression nodes")
@printf("%-24s %8s %14s %12s %10s\n", "layers", "params", "monolithic", "layerwise", "ratio")
for dims in LAYERWISE_ARCHITECTURES
    layerwise = layerwise_nodes(dims)
    if skip_monolithic(dims)
        @printf("%-24s %8d %14s %12d %10s\n", dims, parameterlength(chain(dims)), "skipped",
                layerwise, "-")
    else
        monolithic = monolithic_nodes(dims)
        @printf("%-24s %8d %14d %12d %10.1f\n", dims, parameterlength(chain(dims)), monolithic,
                layerwise, monolithic / layerwise)
    end
end

println("\nSymbolicPullback: what each construction costs to build")
@printf("%-24s %8s %14s %12s\n", "layers", "params", "monolithic (s)", "layerwise (s)")
for dims in LAYERWISE_ARCHITECTURES
    layerwise = measure_pullback(dims, true; layerwise = true)
    if skip_monolithic(dims)
        @printf("%-24s %8d %14s %12.2f\n", dims, parameterlength(chain(dims)), "skipped",
                layerwise.construction)
    else
        monolithic = measure_pullback(dims, true; layerwise = false)
        @printf("%-24s %8d %14.2f %12.2f\n", dims, parameterlength(chain(dims)),
                monolithic.construction, layerwise.construction)
    end
end

if !RUN_EVERYTHING
    println("\nRe-run with `--all` to include the deep networks without CSE (slow: ~440 MB of " *
            "generated code for $(ARCHITECTURES[end])), and the monolithic expressions that were " *
            "skipped above (slower still: the largest has 2·10⁸ nodes).")
end
