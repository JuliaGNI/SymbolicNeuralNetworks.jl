"""
    symbolic_lagrangian is a functions that creates a symbolic lagrangian from any lagrangian.
    The output of the function is 
        - the symbolics parameters used to build the hamiltonian (i.e t, x, v),
        - the symbolic expression of the lagrangian,
        - the function generated from the symbolic lagrangian. 
"""
function symbolic_lagrangian(L::Base.Callable, dim::Int, params::Union{Tuple, NamedTuple, AbstractArray})

    RuntimeGeneratedFunctions.init(@__MODULE__)
    
    # creates variables 
    @variables st                           # for the time
    @variables x(st)[1:dim]               # for the position
    @variables v(st)[1:dim]               # for the velocity
    
    sparams = symbolize(params; redundancy = false)[1]       # for the parameters

    # create the symbolic hamiltonian
    sL = L(st, x, v, sparams)
    
    # create the related code  
    code_L = build_function(sL, st, x, v, develop(sparams)...)

    # rewrite the code to take directly paramters
    rewrite_code_L = rewrite_lagrangian(code_L, (x, v, st), sparams)

    # create the related function
    gL = @RuntimeGeneratedFunction(rewrite_code_L)

    return st, x, v, sparams, sL, gL
end
