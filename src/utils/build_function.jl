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


function build_hamiltonian(H::Base.Callable, dim::Int, params::Union{Tuple, NamedTuple, AbstractArray})

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

function buildsymbolic(nn::NeuralNetwork, dim::Int)

    RuntimeGeneratedFunctions.init(@__MODULE__)

    @variables sinput[1:dim]
    
    sparams = symbolic_params(nn)

    est = nn(sinput, sparams)

    code = build_function(est, sinput, develop(sparams)...)[2]

    rewrite_codes = rewrite_neuralnetwork(code, (sinput,), sparams)

    fun = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(rewrite_codes))

    fun
end
