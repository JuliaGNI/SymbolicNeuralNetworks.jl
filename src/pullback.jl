"""
    SymbolicPullback <: AbstractPullback

`SymbolicPullback` computes the *symbolic pullback* of a loss function.
"""
struct SymbolicPullback{NNLT, FT} <: AbstractPullback{NNLT}
    loss::NNLT
    fun::FT
end

function SymbolicPullback(nn::HamiltonianSymbolicNeuralNetwork)
    loss = HNNLoss(nn)
    symbolic_pullbacks, sinput, soutput = symbolic_pullback(nn, loss)
    pbs_executable = build_executable_gradient(symbolic_pullbacks, sinput, soutput, nn)
    function pbs(input, output, params)
        _ -> (pbs_executable(input, output, params) |> _get_params)
    end
    SymbolicPullback(loss, pbs)
end

_get_params(nt::NamedTuple) = nt
_get_params(ps::NeuralNetworkParameters) = ps.params

# (_pullback::SymbolicPullback)(ps, model, input_nt::QPTOAT)::Tuple = Zygote.pullback(ps -> _pullback.loss(model, ps, input_nt), ps)
function (_pullback::SymbolicPullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT})::Tuple
    _pullback.loss(model, ps, input_nt_output_nt...), _pullback.fun(input_nt_output_nt..., ps)
end

function build_executable_gradient(eqs::Union{NamedTuple, NeuralNetworkParameters}, sinput::Symbolics.Arr, soutput::Symbolics.Arr, sparams::NeuralNetworkParameters)
    vals = Tuple(build_executable_gradient(eqs[key], sinput, soutput, sparams) for key in keys(eqs))
    ps = NamedTuple{keys(eqs)}(vals)
    pbs_executable(ps, input, output, params) = apply_element_wise(ps, input, output, params)
    pbs_executable(input, output, params) = pbs_executable(ps, input, output, params)
    pbs_executable
end

@generated function apply_element_wise(ps::NamedTuple, input, output, params::NeuralNetworkParameters)
    N = length(ps.parameters[1])
    x_symbols = [gensym() for _ in 1:N]
    eqs = [:($x_symbol = ps[$i](input, output, params)) for (x_symbol, i) in zip(x_symbols, 1:N)]
    calls = [eqs..., :(return NamedTuple{$(ps.parameters[1])}(tuple($(x_symbols...))))]
    Expr(:block, calls...)
end

@generated function apply_element_wise(ps::NeuralNetworkParameters, input, output, params::NeuralNetworkParameters)
    N = length(ps.parameters[1])
    x_symbols = [gensym() for _ in 1:N]
    eqs = [:($x_symbol = ps[$i](input, output, params)) for (x_symbol, i) in zip(x_symbols, 1:N)]
    calls = [eqs..., :(return NeuralNetworkParameters{$(ps.parameters[1])}(tuple($(x_symbols...))))]
    Expr(:block, calls...)
end

function build_executable_gradient(eqs, sinput, soutput, nn::AbstractSymbolicNeuralNetwork)
    build_executable_gradient(eqs, sinput, soutput, nn.params)
end

function symbolic_pullback(nn::AbstractSymbolicNeuralNetwork, loss::NetworkLoss)
    input_dim = input_dimension(nn.model)
    output_dim = output_dimension(nn.model)
    @variables sinput[1:input_dim]
    @variables soutput[1:output_dim]

    symbolic_loss = loss(nn.model, nn.params, sinput, soutput)
    symbolic_diffs = symbolic_differentials(nn.params)
    symbolic_gradients = symbolic_gradient(symbolic_loss, symbolic_diffs)
    NeuralNetworkParameters{keys(nn.params)}(symbolic_gradients), sinput, soutput
end

function symbolic_pullback(nn::HamiltonianSymbolicNeuralNetwork, loss::NetworkLoss)
    input_dim = input_dimension(nn.model)
    @variables sinput[1:input_dim]
    @variables soutput[1:input_dim]

    symbolic_loss = loss(nn.model, nn.params, sinput, soutput)
    symbolic_diffs = symbolic_differentials(nn.params)
    symbolic_gradient(symbolic_loss, symbolic_diffs), sinput, soutput
end

function symbolic_differentials(sparams::Symbolics.Arr)
    collect(Differential.(sparams))
end

function symbolic_differentials(sparams::NamedTuple)
    differential_values = (symbolic_differentials(sparams[key]) for key in keys(sparams))
    NamedTuple{keys(sparams)}(differential_values)
end

function symbolic_differentials(sparams::NeuralNetworkParameters)
    vals = Tuple(symbolic_differentials(sparams[key]) for key in keys(sparams))
    NeuralNetworkParameters{keys(sparams)}(vals)
end

function symbolic_gradient(soutput, Dx::AbstractArray)
    [expand_derivatives(Symbolics.scalarize(dx(soutput))) for dx in Dx]
end

function symbolic_gradient(soutput, dps::NamedTuple)
    gradient_values = (symbolic_gradient(soutput, dps[key]) for key in keys(dps))
    NamedTuple{keys(dps)}(gradient_values)
end

function symbolic_gradient(soutput, dps::NeuralNetworkParameters)
    vals = Tuple(symbolic_gradient(soutput, dp) for dp in values(dps))
    NeuralNetworkParameters{keys(dps)}(vals)
end

function build_executable_gradient(eq::EqT, sinput::Symbolics.Arr, soutput::Symbolics.Arr, params::NeuralNetworkParameters)
    gen_fun = _build_executable_gradient(eq, sinput, soutput, params)
    # + here instead of hcat!
    gen_fun_returned(input, output, ps) = mapreduce(k -> gen_fun(input, output, ps, k), +, axes(input, 2))
    gen_fun_returned(input::AT, output::AT, ps) where {AT <: Union{AbstractVector, Symbolics.Arr}} = gen_fun_returned(reshape(input, length(input), 1), reshape(output, length(output), 1), ps)
    gen_fun_returned(input::AT, output::AT, ps) where {T, AT <: AbstractArray{T, 3}} = gen_fun_returned(reshape(input, size(input, 1), size(input, 2) * size(input, 3)), reshape(output, size(output, 1), size(output, 2) * size(output, 3)), ps)
    gen_fun_returned
end

function _build_executable_gradient(eq::EqT, sinput::Symbolics.Arr, soutput::Symbolics.Arr, params::NeuralNetworkParameters)
    code = build_function(eq, sinput, soutput, values(params)...; expression = Val{true}) |> _reduce_code
    rewritten_code = fix_map_reduce(modify_input_arguments2(rewrite_arguments2(fix_create_array(code))))
    parallelized_code = make_kernel2(rewritten_code)
    @RuntimeGeneratedFunction(parallelized_code)
end

"""
    modify_input_arguments2(s)

Change input arguments of type `(sinput, soutput, ps.L1, ps.L2)` etc to `(sinput, soutput, ps)`.
This should be used after [`rewrite_arguments`](@ref).

# Examples

```jldoctest
using SymbolicNeuralNetworks: modify_input_arguments2

s = "(sinput, soutput, ps.L1, ps.L2, ps.L3)"
modify_input_arguments2(s)

# output
"(sinput, soutput, ps)"
```
"""
function modify_input_arguments2(s::AbstractString)
    @assert contains(s, "(sinput, soutput, ") "The first input arguments must be sinput and soutput."
    regex = r"\(sinput, soutput, ps[a-zA-Z0-9., ]+\)"
    replace(s, regex => "(sinput, soutput, ps)")
end

function modify_input_arguments2(expression::Expr)
    Meta.parse(modify_input_arguments2(string(expression)))
end

"""
# Examples
```jldoctest
using SymbolicNeuralNetworks

s = "function (sinput, soutput, ps)\n begin\n getindex(sinput, 1) + getindex(soutput, 2) \n end\n end"
SymbolicNeuralNetworks.make_kernel2(s)

# output

"function (sinput, soutput, ps, k)\n begin\n getindex(sinput, 1, k) + getindex(soutput, 2, k) \n end\n end"
```
"""
function make_kernel2(s::AbstractString)
    # add k to function arguments
    s_added_k = replace(s, "function (sinput, soutput, ps)" => "function (sinput, soutput, ps, k)")
    # add k in body of function
    s_added_k_input = replace(s_added_k, r"getindex\(sinput, ([0-9]+)\)" => s"getindex(sinput, \1, k)")
    replace(s_added_k_input, r"getindex\(soutput, ([0-9]+)\)" => s"getindex(soutput, \1, k)")
end

function make_kernel2(expression::Expr)
    Meta.parse(make_kernel2(string(expression)))
end

function rewrite_arguments2(s::AbstractString)
    regex = r"ˍ₋arg([0-9]+)"
    reformatted = s"ps.L⨸\1⨸"
    expression_with_char = replace(s, regex => reformatted)
    # split at ⨸ symbol:
    expression_split = split(expression_with_char, "⨸")
    *(_modify_integer2.(expression_split)...)
end

function rewrite_arguments2(expression::Expr)
    Meta.parse(rewrite_arguments2(string(expression)))
end

function _modify_integer2(s::AbstractString)
    (contains(s, r"[^0-9]+") || isempty(s)) ? s : "$(Meta.parse(s)-2)"
end