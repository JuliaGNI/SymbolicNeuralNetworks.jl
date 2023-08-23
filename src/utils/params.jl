#=
    This files contains recursive functions to create a preserving shape symbolic params which can be the form of any combinaition of Tuple, namedTuple, Array and Real. 
=#

function SymbolicName(arg, storage; redundancy = true)
    if redundancy
        arg ∈ keys(storage) ? storage[arg] += 1 : storage[arg] = 1
        nam = string(arg)*"_"*string(storage[arg])
        return Symbol(nam)
    else
        nam = string(arg)
        return Symbol(nam)
    end
end

function symbolic_params(::Real, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    sname = ismissing(var_name) ?  SymbolicName(:X, storage; redundancy = redundancy) : SymbolicName(var_name, storage; redundancy = redundancy)
    ((@variables $sname)[1], storage)
end

function symbolic_params(M::AbstractArray, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    sname = ismissing(var_name) ?  SymbolicName(:M, storage; redundancy = redundancy) : SymbolicName(var_name, storage; redundancy = redundancy)
    ((@variables $sname[Tuple([1:s for s in size(M)])...])[1], storage)
end

function symbolic_params(nt::NamedTuple, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    if length(nt) == 1
        symb, storage= symbolic_params(values(nt)[1], keys(nt)[1], storage; redundancy = redundancy)
        return NamedTuple{keys(nt)}((symb,)), storage
    else
        symb, storage = symbolic_params(values(nt)[1], keys(nt)[1], storage; redundancy = redundancy)
        symbs, storage = symbolic_params(NamedTuple{keys(nt)[2:end]}(values(nt)[2:end]), var_name, storage; redundancy = redundancy)
        return  (NamedTuple{keys(nt)}(Tuple([symb, symbs...])), storage)
    end
end

function symbolic_params(t::Tuple, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    if length(t) == 1
        symb, storage = symbolic_params(t[1], var_name, storage; redundancy = redundancy)
        return (symb,), storage
    else
        symb, storage = symbolic_params(t[1], var_name, storage; redundancy = redundancy)
        symbs, storage = symbolic_params(t[2:end], var_name, storage; redundancy = redundancy)
        return (Tuple([symb, symbs...]), storage)
    end
end

symbolic_params(nn::NeuralNetwork) = symbolic_params(nn.params, missing, Dict())[1]