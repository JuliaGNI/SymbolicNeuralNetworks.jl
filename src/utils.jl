function get_string(expr)
    replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
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

