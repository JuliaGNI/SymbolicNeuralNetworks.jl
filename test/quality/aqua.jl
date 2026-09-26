# Aqua.jl quality-assurance checks. See https://github.com/JuliaTesting/Aqua.jl.

using Aqua
using SymbolicNeuralNetworks
using Test

Aqua.test_all(SymbolicNeuralNetworks;
    ambiguities = (broken = true,),     # issue: 5 ambiguities of `evaluate_batch`, not yet filed
    piracies = (broken = true,))        # issue #66
