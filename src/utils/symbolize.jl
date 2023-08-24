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

function symbolize(::Real, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    sname = ismissing(var_name) ?  SymbolicName(:X, storage; redundancy = redundancy) : SymbolicName(var_name, storage; redundancy = redundancy)
    ((@variables $sname)[1], storage)
end

function symbolize(M::AbstractArray, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    sname = ismissing(var_name) ?  SymbolicName(:M, storage; redundancy = redundancy) : SymbolicName(var_name, storage; redundancy = redundancy)
    ((@variables $sname[Tuple([1:s for s in size(M)])...])[1], storage)
end

function symbolize(nt::NamedTuple, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    if length(nt) == 1
        symb, storage= symbolize(values(nt)[1], keys(nt)[1], storage; redundancy = redundancy)
        return NamedTuple{keys(nt)}((symb,)), storage
    else
        symb, storage = symbolize(values(nt)[1], keys(nt)[1], storage; redundancy = redundancy)
        symbs, storage = symbolize(NamedTuple{keys(nt)[2:end]}(values(nt)[2:end]), var_name, storage; redundancy = redundancy)
        return  (NamedTuple{keys(nt)}(Tuple([symb, symbs...])), storage)
    end
end

function symbolize(t::Tuple, var_name::Union{Missing, Symbol} = missing, storage = Dict(); redundancy = true)
    if length(t) == 1
        symb, storage = symbolize(t[1], var_name, storage; redundancy = redundancy)
        return (symb,), storage
    else
        symb, storage = symbolize(t[1], var_name, storage; redundancy = redundancy)
        symbs, storage = symbolize(t[2:end], var_name, storage; redundancy = redundancy)
        return (Tuple([symb, symbs...]), storage)
    end
end

#symbolize(nn::NeuralNetwork) = symbolize(nn.params, missing, Dict())[1]