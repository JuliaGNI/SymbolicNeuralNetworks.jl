using SymbolicNeuralNetworks
using SafeTestsets

@safetestset "Check if reshape works in the correct way with the generated functions.               " begin
    include("reshape_test.jl")
end
@safetestset "Symbolic gradient                                                                      " begin
    include("derivatives/symbolic_gradient.jl")
end
@safetestset "Symbolic Neural network                                                                " begin
    include("derivatives/jacobian.jl")
end
@safetestset "Symbolic Params                                                                        " begin
    include("symbolic_neuralnet/symbolize.jl")
end
@safetestset "Tests associated with 'build_function.jl'                                              " begin
    include("build_function/build_function.jl")
end
@safetestset "Tests associated with 'build_function_double_input.jl'                                 " begin
    include("build_function/build_function_double_input.jl")
end
@safetestset "Tests associated with 'build_function_array.jl                                         " begin
    include("build_function/build_function_arrays.jl")
end
@safetestset "Codegen-drift guard for the Symbolics string pipeline                                  " begin
    include("build_function/codegen_drift.jl")
end
@safetestset "CSE does not change the computed values                                                 " begin
    include("build_function/cse_equivalence.jl")
end
@safetestset "In-place kernels agree with the out-of-place ones                                       " begin
    include("build_function/inplace_equivalence.jl")
end
@safetestset "Joint codegen agrees with per-entry codegen                                             " begin
    include("build_function/joint_codegen.jl")
end
@safetestset "Compare Zygote Pullback with Symbolic Pullback                                         " begin
    include("derivatives/pullback.jl")
end

# Doctests are version-sensitive, so they are opt-in here (and run authoritatively in the
# documentation build, see .github/workflows/Documenter.yml). Enable locally with
# `SYMBOLICNEURALNETWORKS_DOCTESTS=true`.
if get(ENV, "SYMBOLICNEURALNETWORKS_DOCTESTS", "false") == "true"
    @safetestset "Doctests                                                                               " begin
        include("doctest.jl")
    end
end
