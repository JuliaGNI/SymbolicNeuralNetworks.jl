

symbolicparameters(model::Model) = error("symbolicparameters not implemented for model type ", typeof(model))

function symbolicparameters(model::Chain)
    symbolize(Tuple(symbolicparameters(layer) for layer in model))[1]
end


function symbolicparameters(::Dense{M,N,true}) where {M,N}
    @variables W[1:N, 1:M], b[1:N]
    (W = W, b = b)
end

function symbolicparameters(::Dense{M,N,false}) where {M,N}
    @variables W[1:N, 1:M]
    (W = W,)
end

symbolicparameters(nn::NeuralNetwork) = symbolicparameters(model(nn))