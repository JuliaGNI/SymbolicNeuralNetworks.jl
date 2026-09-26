# Aqua.jl quality-assurance checks. See https://github.com/JuliaTesting/Aqua.jl.

using Aqua
using SymbolicNeuralNetworks
using Test

Aqua.test_all(SymbolicNeuralNetworks;
    ambiguities = (broken = true,),     # issue #67
    piracies = (broken = true,))        # issue #66
