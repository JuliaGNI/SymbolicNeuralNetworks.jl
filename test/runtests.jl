using SafeTestsets

@safetestset "Symbolic variables                                                                     " begin
    include("symbolic_neuralnet/symbolic_variables.jl")
end
@safetestset "SymbolicNeuralNetwork                                                                  " begin
    include("symbolic_neuralnet/symbolic_neuralnet.jl")
end
@safetestset "Rewrite rules for the generated code                                                   " begin
    include("codegen/expression_rewriting.jl")
end
@safetestset "Kernels                                                                                " begin
    include("codegen/kernels.jl")
end
@safetestset "build_nn_function                                                                      " begin
    include("codegen/build_nn_function.jl")
end
@safetestset "Batching, allocation and result shapes                                                 " begin
    include("codegen/batched_function.jl")
end
@safetestset "Equation sets                                                                          " begin
    include("codegen/equation_sets.jl")
end
@safetestset "Flat parameters                                                                        " begin
    include("codegen/flat_parameters.jl")
end
@safetestset "Codegen-drift guard                                                                    " begin
    include("codegen/codegen_drift.jl")
end
@safetestset "CSE does not change the computed values                                                " begin
    include("codegen/cse_equivalence.jl")
end
@safetestset "In-place kernels agree with the out-of-place ones                                      " begin
    include("codegen/inplace_equivalence.jl")
end
@safetestset "Generated functions are differentiable                                                 " begin
    include("codegen/zygote_differentiability.jl")
end
@safetestset "Generated functions are type stable                                                    " begin
    include("codegen/type_stability.jl")
end
@safetestset "Jacobian                                                                               " begin
    include("derivatives/jacobian.jl")
end
@safetestset "Gradient                                                                               " begin
    include("derivatives/gradient.jl")
end
@safetestset "SymbolicPullback                                                                       " begin
    include("derivatives/pullback.jl")
end
@safetestset "Layerwise SymbolicPullback                                                              " begin
    include("derivatives/layerwise_pullback.jl")
end

# Doctests are version-sensitive, so they are opt-in here (and run authoritatively in the
# documentation build, see .github/workflows/Documenter.yml). Enable locally with
# `SYMBOLICNEURALNETWORKS_DOCTESTS=true`.
if get(ENV, "SYMBOLICNEURALNETWORKS_DOCTESTS", "false") == "true"
    @safetestset "Doctests                                                                               " begin
        include("doctest.jl")
    end
end
