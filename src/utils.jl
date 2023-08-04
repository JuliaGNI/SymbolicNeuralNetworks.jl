function get_string(expr)
    replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
end

develop(x) = [x]
develop(t::Tuple{Any}) = [develop(t[1])...]
develop(t::Tuple) = [develop(t[1])..., develop(t[2:end])...]
function develop(t::NamedTuple) 
   X = [[develop(e)...] for e in t] 
   vcat(X...)
end


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

