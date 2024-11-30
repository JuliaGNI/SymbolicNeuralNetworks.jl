"""
    SymbolicPullback <: AbstractPullback

`SymbolicPullback` computes the *symbolic pullback* of a loss function.

# Examples

```jldoctest
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
import Random
Random.seed!(123)

c = Chain(Dense(2, 1, tanh))
nn = SymbolicNeuralNetwork(c)
loss = FeedForwardLoss()
pb = SymbolicPullback(nn, loss)
ps = initialparameters(c) |> NeuralNetworkParameters
pv_values = pb(ps, nn.model, (rand(2), rand(1)))[2](1) |> typeof

# output

Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{Float64}, b::Vector{Float64}}}} (alias for Array{@NamedTuple{L1::@NamedTuple{W::Array{Float64, 2}, b::Array{Float64, 1}}}, 1})
```
"""
struct SymbolicPullback{NNLT, FT} <: AbstractPullback{NNLT}
    loss::NNLT
    fun::FT
end

function SymbolicPullback(nn::HamiltonianSymbolicNeuralNetwork)
    SymbolicPullback(nn, HNNLoss(nn))
end

function SymbolicPullback(nn::SymbolicNeuralNetwork, loss::NetworkLoss)
    @variables soutput[1:output_dimension(nn.model)]
    symbolic_loss = loss(nn.model, nn.params, nn.input, soutput)
    symbolic_pullbacks = symbolic_pullback(symbolic_loss, nn)
    pbs_executable = build_nn_function(symbolic_pullbacks, nn.params, nn.input, soutput)
    function pbs(input, output, params)
        _ -> (pbs_executable(input, output, params) |> _get_params)
    end
    SymbolicPullback(loss, pbs)
end

SymbolicPullback(nn::SymbolicNeuralNetwork) = SymbolicPullback(nn, AbstractNeuralNetworks.FeedForwardLoss())

_get_params(nt::NamedTuple) = nt
_get_params(ps::NeuralNetworkParameters) = ps.params
_get_params(ps::AbstractArray{<:Union{NamedTuple, NeuralNetworkParameters}}) = [_get_params(nt) for nt in ps]

# (_pullback::SymbolicPullback)(ps, model, input_nt::QPTOAT)::Tuple = Zygote.pullback(ps -> _pullback.loss(model, ps, input_nt), ps)
function (_pullback::SymbolicPullback)(ps, model, input_nt_output_nt::Tuple{<:QPTOAT, <:QPTOAT})::Tuple
    _pullback.loss(model, ps, input_nt_output_nt...), _pullback.fun(input_nt_output_nt..., ps)
end