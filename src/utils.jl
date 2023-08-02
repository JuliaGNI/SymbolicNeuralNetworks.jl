function get_string(expr)
    replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
end

develop(x) = [x]
develop(t::Tuple{Any}) = [develop(t[1])...]
develop(t::Tuple) = [develop(t[1])..., develop(t[2:end])...]
develop(t::NamedTuple) = vcat([[develop(e)...] for e in t]...)


function transposymplecticMatrix(n::Int) 
    I = Diagonal(ones(n÷2))
    Z = zeros(n÷2,n÷2)
    [Z -I; I Z]
end

function symplecticMatrix(n::Int) 
    I = Diagonal(ones(n÷2))
    Z = zeros(n÷2,n÷2)
    [Z I; -I Z]
end

#=
using Zygote

(::Zygote.ProjectTo{Float64})(x::Tuple{Float64}) = only(x)

(::Zygote.ProjectTo{AbstractArray})(x::Tuple{Vararg{Float64}}) = [x...]

dev(x) = Zygote.gradient(x->sum(develop(x)), x)[1]

dev(4)
dev((4,4))
dev((1,(1,(1,1))))

Base.size(nt::NamedTuple) = (length(nt),)

dev((1,(W=2, b=5)))
=#