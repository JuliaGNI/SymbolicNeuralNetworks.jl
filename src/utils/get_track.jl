#=
    This file contains functions to rewrite the expression of the function built by Symbolics.jl. 
=#

function get_track(W, SW, s = nothing)
    return W===SW ? (true, s) : (false, nothing)
end

function get_track(t::Tuple, W, s::String, info = 1)
    if length(t) == 1
        return get_track(t[1], W, string(s,"[",info,"]"))
    else
        bool, str = get_track(t[1], W, string(s,"[",info,"]"))
        return bool ? (true, str) : get_track(t[2:end], W, s, info+1)
    end
end

function get_track(nt::NamedTuple, W, s::String)
    for k in keys(nt)
        bool, str = get_track(nt[k], W, string(s,".",k))
        if bool
            return (bool, str)
        end
    end
    (false, nothing)
end
