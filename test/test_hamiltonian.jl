using SymbolicNeuralNetworks
using Symbolics
using Test

parameters = (k=1, )

function hamiltonian(t, q, p, params)
    p[1]^2 / 2 + params.k * q[1]^2 / 2
end

st̃, sq̃, sp̃, sparams, shamiltonian, hamiltonian_function = symbolic_hamiltonian(hamiltonian, 2, parameters)

@variables st
@variables q(st)[1:1]
@variables p(st)[1:1]
@variables k

@test isequal(st, st̃)
@test isequal(q, sq̃)
@test isequal(p, sp̃)
@test isequal(sparams, (k = k,))

@test isequal(shamiltonian, Num((1//2)*(p[1]^2) + (1//2)*k*(q[1]^2)))

t, q, p = 2, 0.5, 0.7
@test hamiltonian_function(t, q, p, parameters) == hamiltonian(t, q, p, parameters)

