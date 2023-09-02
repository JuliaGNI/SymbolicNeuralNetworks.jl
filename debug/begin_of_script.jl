using Symbolics
using GeometricMachineLearning
using AbstractNeuralNetworks
import AbstractNeuralNetworks: layer, AbstractLayer
using Base.Iterators
using SparseArrays

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

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

# Symbolize layer

function symbolize(l::AbstractLayer)
    sparams = symbolicparameters(l)
    sin = size_input(l)
    @variables x[1:sin]
    l(x, sparams)
end

function symbolize_eachlayer(chain::Chain)
    symbolic_layers = []
    index_layer = zeros(Int64, length(chain))
    memo = Dict()
    for (l,i) in zip(model(hnn), eachindex(chain))
        stl = Symbol(typeof(l))
        if stl ∉ keys(memo)
            sl = symbolize(l)
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

symbolic_layers, index_layer, memo =  symbolize_eachlayer(model(hnn))


function symbolics_computation(bricks::AbstractVector)
    sbricks = []
    for b in bricks

    end
end


function compute_one_derivative(chain, symbolic_layers, index_layer, idxi)
    memo = Dict()
    IndexSparse = Vector{Int64}[[idxi]]
    for (n,m) in zip(eachindex(index_layer),index_layer)
        ln = layer(chain,n)
        szin = size_input(ln)
        szout = size_output(ln)
        stl = Symbol(typeof(ln))
        if stl ∉ keys(memo)
            @variables x[1:szin]
            indexR, indexC = unzip([collect(Iterators.product(1:szout,IndexSparse[end]))...]) 
            Jacln = Symbolics.sparsejacobian_vals(symbolic_layers[m], x, indexR, indexC)
            IndexSparse_n = 1:4
            push!(IndexSparse, IndexSparse_n)
            s = spzeros(eltype(Jacln), szout, szin)
            for (i,j,k) in zip(indexR,indexC, eachindex(Jacln))
                s[i,j] = Jacln[k]
            end
            memo[stl] = s
        end
    end
    NamedTuple(memo), IndexSparse
end



memo2, IndexSparse = compute_one_derivative(model(hnn), symbolic_layers, index_layer, 1)