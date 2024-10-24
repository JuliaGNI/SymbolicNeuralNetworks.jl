"""
    symbolic_hamitonian is a functions that creates a symbolic hamiltonian from any hamiltonian.
    The output of the function is 
        - the symbolics parameters used to build the hamiltonian (i.e t, q, p),
        - the symbolic expression of the hamiltonian,
        - the function generated from the symbolic hamiltonian. 
"""
function symbolic_hamiltonian(H::Base.Callable, dim::Int, params::Union{Tuple, NamedTuple, AbstractArray})

    RuntimeGeneratedFunctions.init(@__MODULE__)

    # creates variables 
    @variables st                         # for the time
    @variables q(st)[1:dim]               # for the position
    @variables p(st)[1:dim]               # for the momentum
    
    sparams = symbolize(params; redundancy = false)[1]        # for the parameters

    # create the symbolic hamiltonian
    sH = H(st, q, p, sparams)
    
    # create the related code  
    code_H = build_function(sH, st, q, p, develop(sparams)...)

    # rewrite the code to take directly paramters
    rewrite_code_H = rewrite_hamiltonian(code_H, (q, p, st), sparams)

    # create the related function
    gH = @RuntimeGeneratedFunction(rewrite_code_H)

    return st, q, p, sparams, sH, gH
end
