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
    # get symbolic input variables with specified dimension
    @variables sinput[1:dim]
    
    # get symbolic representation of parameters
    sparams = symbolic_params(nn)

    # evaluate network on symbolic inputs and parameters
    snn = nn(sinput, sparams)

    # generate code for evaluating the network
    code = build_function(snn, sinput, develop(sparams)...)[2]

    # rewrite function signatures and function names
    rewrite_codes = rewrite_neuralnetwork(code, (sinput,), sparams)

    # inject code into current module
    @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(rewrite_codes))
end
