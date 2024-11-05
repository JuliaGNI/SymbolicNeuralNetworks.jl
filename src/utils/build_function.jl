#=
     This files contains general functions to create symbolics functions.
=#

function build_eval(f::Base.Callable, args...; params = params::Union{Tuple, NamedTuple, AbstractArray})
    # create symbolic variables for the arguments and parameters
    sargs = symbolic_params(args)
    sparams = symbolic_params(params)

    # create symbolic neural network
    snn = f(sargs..., sparams)

    build_function(snn, develop(sargs)..., develop(sparams)...)[2]
end


function build_hamiltonian(H::Base.Callable, dim::Int, params::Union{Tuple, NamedTuple, AbstractArray})
        # compute the symplectic matrix
        sympmatrix = symplecticMatrix(dim)
        
        # create symbolic variables 
        @variables sq[1:dim÷2]               # position
        @variables sp[1:dim÷2]               # momentum
        @variables st                        # time
        sparams = symbolic_params(params)    # parameters

        # create symbolic Hamiltonian neural network
        snn = H(st, sq, sp, sparams)

        # compute the vectorfield from the hamiltonian
        field = sympmatrix * Symbolics.gradient(snn, [sq..., sp...])
    
        fun_snn = build_function(snn, st, sq, sp, develop(sparams)...)[2]
        fun_field = build_function(field, st, sq, sp, develop(sparams)...)[1]
    
        return (fun_snn, fun_field)
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
