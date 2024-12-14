using AbstractNeuralNetworks: NeuralNetworkParameters
using SymbolicNeuralNetworks
using SymbolicNeuralNetworks: symbolize!
using Symbolics
using Test

# a tuple of `NamedTuple`s and `Tuple`s.
params = NeuralNetworkParameters(  
        (L1 = (W = [1,1], b = [2, 2]),
         L2 = (W = (a = 4 , b = [1,2]), c = 2),
         L3 = [4 5; 7 8],
         L4 = (a = 7, b = 8.2)
        ) )

cache = Dict()
sparams = symbolize!(cache, params, :W)

@variables W_1[Base.OneTo(2)] W_3 W_4[Base.OneTo(2)] W_2[Base.OneTo(2)] W_5 W_6[Base.OneTo(2), Base.OneTo(2)] W_7 W_8

verified_sparams = NeuralNetworkParameters(
        (L1 = (W = W_1, b = W_2), 
         L2 = (W = (a = W_3 , b = W_4), c = W_5), 
         L3 = W_6, 
         L4 = (a = W_7, b = W_8)
         ) )

@test sparams === verified_sparams