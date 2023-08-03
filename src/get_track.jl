#=
    This file contains functions to rewrite the expression of the function built by Symbolics.jl. 
=#


function get_track(W, SW, s)
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


function rewrite(fun, SV)
    for e in develop(SV)
        str_symbol = replace(string(e), r"\[.*"=>"")
        track = get_track(SV, e, "nt")[2]
        fun = Meta.parse(replace(string(fun), str_symbol => track))
    end
    fun = Meta.parse(replace(string(fun), "SX" => "X"))
    fun = Meta.parse(replace(string(fun), r"function .*" => "function (sinput, nt)\n"))
end

#=
@variables W1[1:2,1:2] W2[1:2,1:2] b1[1:2] b2[1:2]
SV = ((W = W1, b = b1), (W = W2, b = b2))


develop(x) = [x]
develop(t::Tuple{Any}) = [develop(t[1])...]
develop(t::Tuple) = [develop(t[1])..., develop(t[2:end])...]
function develop(t::NamedTuple) 
   X = [[develop(e)...] for e in t] 
   vcat(X...)
end

z = SV[2].W  * tanh.(SV[1].W * SX + SV[1].b) + SV[2].b

fun = build_function(z, SX, develop(SV)...)[2]



f= eval(rewrite(fun,SV))
V = ((W = [1 3; 2 2], b = [1, 0]), (W = [1 1; 0 2], b = [1, 0]))
X = [1, 0.2]
f(X, V) 
=#


#=
using GeometricMachineLearning
include("symbolic_params.jl")
nn = NeuralNetwork(HamiltonianNeuralNetwork(2), Float64)
@variables sargs[1:2]         
sparams = symbolic_params(nn.params)[1]    
est = nn(sargs, sparams)
fun = build_function(est, sargs, develop(sparams)...)[2]
=#