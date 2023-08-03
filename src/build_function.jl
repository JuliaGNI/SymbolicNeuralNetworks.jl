#=
     This files contains general functions to create symbolics functions.
=#

function build_eval(f::Base.Callable, args...; params = params::Union{Tuple, NamedTuple, AbstractArray})

    # creates variables 
    sargs = symbolic_params(args)             # for the argument
    sparams = symbolic_params(params)    # for the parameters

    # create the estimation of the hamiltonian
    est = f(sargs..., sparams)

    build_function(est, develop(sargs)..., develop(sparams)...)[2]
end


function build_hamiltonien(H::Base.Callable, dim::Int, params::Union{Tuple, NamedTuple, AbstractArray})

        #compute the symplectic matrix
        sympmatrix = symplecticMatrix(dim)
        
        # creates variables 
        @variables sq[1:dim÷2]               # for the position
        @variables sp[1:dim÷2]               # for the momentum
        @variables st                        # for the time
        sparams = symbolic_params(params)     # for the parameters

        # create the estimation of the hamiltonian
        est = H(st, sq, sp, sparams)

        # compute the vectorfield from the hamiltonian
        field =  sympmatrix * Symbolics.gradient(est, [sq..., sp...])
    
        fun_est = build_function(est, st, sq, sp, develop(sparams)...)[2]
        fun_field = build_function(field,  st, sq, sp, develop(sparams)...)[1]
    
        return (fun_est, fun_field)
end

#=
function build_gradloss(loss::Base.Callable, params::Union{Tuple, NamedTuple, AbstractArray})

    # creates variables 
    sparams = symbolicParams(params)     # for the parameters

    # create the estimation of the hamiltonian
    sloss = loss(sparams)

    # compute the vectorfield from the hamiltonian
    gradloss= Symbolics.gradient(sloss , sparams)

    build_function(gradloss, st, sq, sp, develop(sparams)...)[2]
end
=#