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
#
# Run with
#     julia --project=. scripts/codegen_comparison.jl
# Add `--all` to also generate the deep networks *without* CSE. Be warned: for 5-10-10-10-1 that
# emits roughly 440 MB of code and takes minutes, which is exactly the point being made.

using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: generated_expression, parameter_arguments, symbolic_parameter_gradient
using AbstractNeuralNetworks
using AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, FeedForwardLoss, params, output_dimension
using Symbolics
using Printf
import Random

Random.seed!(123)

const RUN_EVERYTHING = "--all" ∈ ARGS
const ARCHITECTURES = ((5, 10, 1), (5, 10, 10, 1), (5, 10, 10, 10, 1))
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
function measure_pullback(dims, cse::Bool)
    c = chain(dims)
    snn = SymbolicNeuralNetwork(c)
    nn = NeuralNetwork(c)
    ps = params(nn)

    construction = @elapsed pb = SymbolicPullback(snn, FeedForwardLoss(); cse = cse)

    input = rand(dims[1], BATCH_SIZE)
    output = rand(dims[end], BATCH_SIZE)
    evaluate() = pb(ps, c, (input, output))[2](1.0)
    evaluate()  # compile the generated function before timing it
    seconds = minimum(@elapsed(evaluate()) for _ in 1:10)
    (; construction, seconds, bytes = @allocated evaluate())
end

println("Julia $(VERSION), $(BATCH_SIZE)-column batches\n")

# compile `build_function` and the pullback machinery before anything is timed
measure_codegen(ARCHITECTURES[1], true)
measure_pullback(ARCHITECTURES[1], true)

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
    result = measure_pullback(dims, true)
    @printf("%-20s %12.2f %12.3f %12.1f\n", dims, result.construction, result.seconds * 1e3, result.bytes / 1024)
end

if !RUN_EVERYTHING
    println("\nRe-run with `--all` to include the deep networks without CSE (slow: ~440 MB of " *
            "generated code for $(ARCHITECTURES[end])).")
end
