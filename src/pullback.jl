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