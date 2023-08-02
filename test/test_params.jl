using SymbolicNeuralNetworks
using Symbolics
using Test


params = ( (W = [1,1], b = [2, 2]),  (W = (4 , [1,2]), c = 2), [4 5; 7 8], (7, 8.2))

sparams = symbolic_params(params)[1]

@variables W_1[1:2] W_2 W_3[1:2] b_1[1:2] c_1 M_1[1:2, 1:2] X_1 X_2

verified_sparams = ( (W = W_1, b = b_1), (W = (W_2 , W_3), c = c_1), M_1, (X_1, X_2))

@test sparams === verified_sparams