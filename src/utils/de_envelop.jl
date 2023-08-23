#=
    The functions develop creates a n enumerating vector of arrays or reals (can be both) from any structure made with any combinaison of named tuples ans tuple ending in arrays or reals.
    The option completely enables or not to iterates the elements of an array, if not the array is considered as a final element.
=#

develop(x; completely = false) = return completely ? x : [x]
develop(t::Tuple{Any}; completely = false) = [develop(t[1]; completely = completely)...]
develop(t::Tuple; completely = false) = [develop(t[1]; completely = completely)..., develop(t[2:end]; completely = completely)...]
develop(t::NamedTuple; completely = false) = vcat([[develop(e; completely = completely)...] for e in t]...)

#=
    The functions evelop is the reverse functions of develop.
=#

function envelop(::Any, X; kwars...)
    X[1], X[2:end]
end

function envelop(M::AbstractArray, X; completely = false)
    return completely ? (reshape(X[1:length(M)], size(M)), X[length(M)+1:end]) : (X[1], X[2:end])
end

function envelop(t::Tuple, X; completely = false)
    if length(t) == 1
        r, Y = envelop(t[1], X; completely = completely)
        return (r,), Y
    else
        r, Y = envelop(t[1], X; completely = completely)
        rr, Z = envelop(t[2:end], Y; completely = completely)
        return (r, rr...), Z
    end
end

function envelop(nt::NamedTuple, X; completely = false)
    if length(nt) == 1
        r, Y = envelop(nt[1], X; completely = completely)
        return NamedTuple{keys(nt)}((r,)), Y
    else
        r, Y = envelop(nt[1], X; completely = completely)
        rr, Z = envelop(NamedTuple{keys(nt)[2:end]}(values(nt)[2:end]), Y; completely = completely)
        return NamedTuple{keys(nt)}((r, rr...)), Z
    end
end



