using SymbolicNeuralNetworks
using Test

S = ((W=[1,2], b = 7), ((4,5), 7))
dS = develop(S)
Es = envelop(S, dS)[1]

@test dS == [[1,2], 7, 4, 5, 7]
@test S == Es

dSc = develop(S; completely = true)
Esc = envelop(S, dSc; completely = true)[1]

@test dSc == [1,2, 7, 4, 5, 7]
@test S == Esc