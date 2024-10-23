using SymbolicNeuralNetworks
using Symbolics
using Test

parameters = (k=1, )

function lagrangian(t, q, v, params)
    v[1]^2 / 2 - params.k * q[1]^2 / 2
end

st̃, sq̃, sp̃, sparams, slagrangian, lagrangian_function = symbolic_lagrangian(lagrangian, 2, parameters)

@variables st
@variables x(st)[1:1]
@variables v(st)[1:1]
@variables k

@test isequal(st, st̃)
@test isequal(x, sq̃)
@test isequal(v, sp̃)
@test isequal(sparams, (k = k,))

@test isequal(slagrangian, Num((1//2)*(v[1]^2) - (1//2)*k*(x[1]^2)))

t, q, v = 2, 0.5, 0.7
@test lagrangian_function(t, q, v, parameters) == lagrangian(t, q, v, parameters)

