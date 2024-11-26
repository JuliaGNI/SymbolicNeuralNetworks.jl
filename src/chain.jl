function symbolicparameters(model::Chain)
    vals = symbolize(Tuple(symbolicparameters(layer) for layer in model))[1]
    keys = Tuple(Symbol("L$(i)") for i in 1:length(vals))
    NeuralNetworkParameters(NamedTuple{keys}(vals))
end