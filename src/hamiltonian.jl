"""
    HamiltonianSymbolicNeuralNetwork

symbolic_hamitonian is a functions that creates a symbolic hamiltonian from any hamiltonian.
The output of the function is 
- the symbolics parameters used to build the hamiltonian (i.e t, q, p),
- the symbolic expression of the hamiltonian,
- the function generated from the symbolic hamiltonian. 
"""
function HamiltonianSymbolicNeuralNetwork(arch::Architecture, model::Model)

    input_dim = input_dimension(model)
    n = input_dim ÷ 2
    @assert iseven(input_dim) "Input dimension has to be an even number."
    @assert output_dimension(model) == 1 "Output dimension of network has to be scalar."
     
    x = Symbolics.variables(:x, 1:input_dim)
    ∇nn = Symbolics.variables(:∇nn, 1:input_dim)
    nn = Symbolics.variables(:nn, 1:1)

    # placeholder for one
    @variables o 
    o_vec = repeat([o], n)
    𝕀 = Diagonal(o_vec)
    𝕆 = zero(𝕀)
    𝕁 = hcat(vcat(𝕆, -𝕀), vcat(𝕀, 𝕆))
    eqs = (x = x, nn = nn, ∇nn = ∇nn, hvf = substitute(𝕁 * ∇nn, Dict(o => 1, )))

    SymbolicNeuralNetwork(arch, model; eqs = eqs)
end

HamiltonianSymbolicNeuralNetwork(model::Model) = HamiltonianSymbolicNeuralNetwork(UnknownArchitecture(), model)