using Symbolics
using GeometricMachineLearning
using SymbolicNeuralNetworks
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

function SymbolicNeuralNetworks.symbolicparameters(d::Gradient{M, N, true}) where {M,N}
    @variables K[1:d.second_dim÷2, 1:M÷2]
    @variables b[1:d.second_dim÷2]
    @variables a[1:d.second_dim÷2]
    (weight = K, bias = b, scale = a)
end

function SymbolicNeuralNetworks.symbolicparameters(d::Gradient{M, N, false}) where {M,N}
    @variables a[1:d.second_dim÷2, 1:1]
    (scale = a, )
end

arch = GSympNet(2; nhidden = 1, width = 4, allow_fast_activation = false)
sympnet=NeuralNetwork(arch, Float64)

ssympnet = SymbolicNeuralNetwork(arch, 2)

eva = equations(ssympnet).eval

@variables x[1:2]
sparams = symbolicparameters(model(ssympnet))

code = build_function(eva, x, sparams...)[1]

postcode = SymbolicNeuralNetworks.rewrite_neuralnetwork(code, (x,), sparams)

x = [1,2]

H = functions(ssympnet).eval

expr = :(function (x::AbstractArray, p::Tuple)
    (((adjoint(p[1].weight) * p[1].scale .* tanh.(p[1].weight * x[2] + p[1].bias))[1] + x[1])[1], x[2])
end)
#=
SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), Val{(2,)}(), getindex(broadcast(+, Real[x[1]], adjoint((params[1]).weight) * broadcast(*, (params[1]).scale, broadcast(tanh, broadcast(+, (params[1]).weight * Real[x[2]], (params[1]).bias)))), 1), getindex(x, 2))
=#

funnn = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(expr))
funnn(x, sympnet.params)

@time functions(ssympnet).eval(x, sympnet.params)
@time funnn(x, sympnet.params)
@time sympnet(x, sympnet.params)