
function optimize_code!(expr)
    try expr.args
    catch
        return expr
    end
    for i in eachindex(expr.args)
        expr.args[i] =  optimize_code!(expr.args[i])
    end
    if expr.args[1] == :broadcast
        if length(expr.args) == 4
            return :(($(expr.args[2])).($(expr.args[3]), $(expr.args[4])))
        elseif length(expr.args) == 3
            return :(($(expr.args[2])).($(expr.args[3])))
        end
    elseif expr.args[1] == :getindex
        return Meta.parse(string(expr.args[2],"[",expr.args[3],"]"))
    elseif expr.args[1] == :Real
        return expr.args[2]
    end
    return expr
end