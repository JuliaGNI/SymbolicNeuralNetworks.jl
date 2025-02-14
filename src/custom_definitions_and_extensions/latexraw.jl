function _latexraw(args::AbstractNeuralNetworks.GenericActivation; kwargs...)
    _latexraw(args.σ; kwargs...)
end

function _latexraw(args::AbstractNeuralNetworks.TanhActivation; kwargs...)
    _latexraw(tanh; kwargs...)
end