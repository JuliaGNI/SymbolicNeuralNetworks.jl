#=
    This files contains code to rewrite a function under the shape of an expression to modify the input of the function.
=#

function rewrite_code(expr, sargs, sparams = nothing, fun_name = "")
    expr = Meta.parse(replace(string(expr), "var" => ""))
    expr = Meta.parse(replace(string(expr), r"\""=>""))
    for arg in sargs
        str_symbol = replace(string(arg), r"\[.*"=>"")
        if str_symbol[1] == '(' && str_symbol[end] == ')'
            str_symbol = str_symbol[2:end-1]
        end
        track = get_track(sargs, arg, "args")[2]
        expr = Meta.parse(replace(string(expr), str_symbol => track))
    end
    for p in develop(sparams)
        sparams
        p
        str_symbol = replace(string(p), r"\[.*"=>"")
        track = get_track(sparams, p, "params")[2]
        expr = Meta.parse(replace(string(expr), str_symbol => track))
    end
    expr = Meta.parse(replace(string(expr), r"function.*" => "function "*fun_name*"(args, params)\n"))
end

function rewrite_hamiltonian(expr, args, sparams = nothing, fun_name = "shamiltonian")
    expr = rewrite_code(expr, args, sparams, fun_name)
    string_expr = replace(string(string_expr), "args[1]" => "q")
    string_expr = replace(string(string_expr), "args[2]" => "p")
    string_expr = replace(string(string_expr), "args[3]" => "t")
    string_expr = replace(string(string_expr), "args" => "p, t, q")
    expr = Meta.parse(string_expr)
end

function rewrite_lagrangian(expr, args, sparams = nothing, fun_name = "slagrangian")
    expr = rewrite_code(expr, args, sparams, fun_name)
    string_expr = replace(string(expr), "args[1]" => "x")
    string_expr = replace(string(string_expr), "args[2]" => "v")
    string_expr = replace(string(string_expr), "args[3]" => "t")
    string_expr = replace(string(string_expr), "args" => "t, x, v")
    expr = Meta.parse(string_expr)
end

