#=
    This files contains code to rewrite a function under the shape of an expression to modify the input of the function.
=#

function rewrite_code(expr, sargs, sparams = nothing, fun_name = "")
    expr_str = string(expr)
    expr_str = replace(expr_str, "var" => "")
    expr_str = replace(expr_str, r"\"" => "")
    for arg in sargs
        str_symbol = replace(string(arg), r"\[.*" => "")
        if str_symbol[1] == '(' && str_symbol[end] == ')'
            str_symbol = str_symbol[2:end-1]
        end
        track = get_track(sargs, arg, "args")[2]
        expr_str = replace(expr_str, str_symbol => track)
    end
    for p in develop(sparams)
        str_symbol = replace(string(p), r"\[.*" => "")
        track = get_track(sparams, p, "params")[2]
        expr_str = replace(expr_str, str_symbol => track)
    end
    expr_str = replace(expr_str, r"function.*" => "function " * fun_name * "(args, params)\n")
    expr = Meta.parse(expr_str)
end

function rewrite_hamiltonian(expr, args, sparams = nothing, fun_name = "shamiltonian")
    expr = rewrite_code(expr, args, sparams, fun_name)
    string_expr = string(expr)
    string_expr = replace(string_expr, "args[1]" => "q")
    string_expr = replace(string_expr, "args[2]" => "p")
    string_expr = replace(string_expr, "args[3]" => "t")
    string_expr = replace(string_expr, "args" => "t, q, p")
    expr = Meta.parse(string_expr)
end

function rewrite_lagrangian(expr, args, sparams = nothing, fun_name = "slagrangian")
    expr = rewrite_code(expr, args, sparams, fun_name)
    string_expr = string(expr)
    string_expr = replace(string_expr, "args[1]" => "x")
    string_expr = replace(string_expr, "args[2]" => "v")
    string_expr = replace(string_expr, "args[3]" => "t")
    string_expr = replace(string_expr, "args" => "t, x, v")
    expr = Meta.parse(string_expr)
end

function rewrite_neuralnetwork(expr, args, sparams)
    expr = rewrite_code(expr, args, sparams)
    string_expr = string(expr)
    string_expr = replace(string_expr, "args[1]" => "x")
    string_expr = replace(string_expr, "args" => "x")
    expr = Meta.parse(string_expr)
end
