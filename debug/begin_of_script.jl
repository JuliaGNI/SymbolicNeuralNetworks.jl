using Symbolics
using GeometricMachineLearning
using AbstractNeuralNetworks
import AbstractNeuralNetworks: layer

function symbolicparameters(::Dense{M,N,true}) where {M,N}
    @variables W[1:N, 1:M], b[1:N]
    (W = W, b = b)
end

function symbolicparameters(::Dense{M,N,false}) where {M,N}
    @variables W[1:N, 1:M]
    (W = W,)
end

size_input(::Dense{M,N}) where {M,N} = M
size_output(::Dense{M,N}) where {M,N} = N

arch = HamiltonianNeuralNetwork(2; nhidden = 7, width = 4)
hnn = NeuralNetwork(arch, Float16)


function symbolize_layers(chain)
    symbolic_layers = []
    index_layer = zeros(Int64, length(chain))
    memo = Dict()
    for (l,i) in zip(model(hnn), eachindex(chain))
        stl = Symbol(typeof(l))
        if stl ∉ keys(memo)
            sparams = symbolicparameters(l)
            sin = size_input(l)
            @variables x[1:sin]
            sl = l(x, sparams)
            push!(symbolic_layers, sl)
            size_sl = length(symbolic_layers)
            index_layer[i] = size_sl
            memo[stl] = size_sl
        else
            ind = memo[stl]
            index_layer[i] = only(ind)
        end
    end
    symbolic_layers, index_layer, NamedTuple(memo)
end



symbolic_layers, index_layer, memo =  symbolize_layers(model(hnn))
@show symbolic_layers
@show index_layer
@show memo


function compute_one_derivative(chain, symbolic_layers, index_layer, idxi)
 
    IndexSparse = Vector{Vector{Int16}}[]
    l1 = layer(chain,1)
    push!(IndexSparse, 1:size_output(l1))
    for n in eachindex(index_layer)
        ln = layer(chain,1)
        IndexSparse_n = Vector{Int16}[]
        for k in IndexSparse[]
        @variables x[1:size_input(l)]
        Jacln = sparsejacobian_vals(symbolic_layers[n], x, I, J)

        end
    end
end