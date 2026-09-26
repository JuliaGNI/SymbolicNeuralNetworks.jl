using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Symbolic variables" include("symbolic_neuralnet/symbolic_variables.jl")
    @safetestset "SymbolicNeuralNetwork" include("symbolic_neuralnet/symbolic_neuralnet.jl")
    @safetestset "Rewrite rules for the generated code" include("codegen/expression_rewriting.jl")
    @safetestset "Kernels" include("codegen/kernels.jl")
    @safetestset "build_nn_function" include("codegen/build_nn_function.jl")
    @safetestset "Batching, allocation and result shapes" include("codegen/batched_function.jl")
    @safetestset "Equation sets" include("codegen/equation_sets.jl")
    @safetestset "Flat parameters" include("codegen/flat_parameters.jl")
    @safetestset "Codegen-drift guard" include("codegen/codegen_drift.jl")
    @safetestset "CSE does not change the computed values" include("codegen/cse_equivalence.jl")
    @safetestset "In-place kernels agree with the out-of-place ones" include("codegen/inplace_equivalence.jl")
    @safetestset "Generated functions are differentiable" include("codegen/zygote_differentiability.jl")
    @safetestset "Generated functions are type stable" include("codegen/type_stability.jl")
    @safetestset "Generated functions do not allocate more than they must" include("codegen/allocations.jl")
    @safetestset "Jacobian" include("derivatives/jacobian.jl")
    @safetestset "Gradient" include("derivatives/gradient.jl")
    @safetestset "SymbolicPullback" include("derivatives/pullback.jl")
    @safetestset "Layerwise SymbolicPullback" include("derivatives/layerwise_pullback.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
