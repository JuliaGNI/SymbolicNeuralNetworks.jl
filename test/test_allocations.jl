using Symbolics
using GeometricMachineLearning
using SymbolicNeuralNetworks
using RuntimeGeneratedFunctions
#using BenchmarkTools

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
sympnet = NeuralNetwork(arch, Float64)

ssympnet = SymbolicNeuralNetwork(arch, 2)

eva = equations(ssympnet).eval

@variables x[1:2]
sparams = symbolicparameters(model(ssympnet))

code = build_function(eva, x, sparams...)[1]

postcode = SymbolicNeuralNetworks.rewrite_neuralnetwork(code, (x,), sparams)

x = [1,2]

H = functions(ssympnet).eval

expr = :(function (x::AbstractArray, params::Tuple)
    SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), Val{(2,)}(), getindex(broadcast(+, Real[x[1]], adjoint((params[1]).weight) * broadcast(*, (params[1]).scale, broadcast(tanh, broadcast(+, (params[1]).weight * Real[x[2]], (params[1]).bias)))), 1), getindex(x, 2))
end)
#=
SymbolicUtils.Code.create_array(Array, nothing, Val{1}(), Val{(2,)}(), getindex(broadcast(+, Real[x[1]], adjoint((params[1]).weight) * broadcast(*, (params[1]).scale, broadcast(tanh, broadcast(+, (params[1]).weight * Real[x[2]], (params[1]).bias)))), 1), getindex(x, 2))
=#

expr2 = :(function (x::AbstractArray, p::Tuple)
    getindex(broadcast(+, x[1], adjoint((p[1]).weight) * broadcast(*, (p[1]).scale, broadcast(tanh, broadcast(+, (p[1]).weight * x[2], (p[1]).bias)))), 1), getindex(x, 2)
end)

fun1 = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(expr))
fun1(x, sympnet.params)

fun2 = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(expr2))
fun2(x, sympnet.params)

sympnet(x, sympnet.params)

@time functions(ssympnet).eval(x, sympnet.params)
@time fun1(x, sympnet.params)
@time fun2(x, sympnet.params)
@time sympnet(x, sympnet.params)

#=
@benchmark functions(ssympnet).eval(x, sympnet.params)
@benchmark fun1(x, sympnet.params)
@benchmark fun2(x, sympnet.params)
@benchmark sympnet(x, sympnet.params)
=#

function optimize_code!(expr)
    try expr.args
    catch
        return expr
    end
    for i in eachindex(expr.args)
        expr.args[i] =  optimize_code!(expr.args[i])
    end
    if expr.args[1] == :broadcast
        if length(expr.args) == 4
            return :(($(expr.args[2])).($(expr.args[3]), $(expr.args[4])))
        elseif length(expr.args) == 3
            return :(($(expr.args[2])).($(expr.args[3])))
        end
    elseif expr.args[1] == :getindex
        return Meta.parse(string(expr.args[2],"[",expr.args[3],"]"))
    elseif expr.args[1] == :Real
        return expr.args[2]
    end
    return expr
end

expr = optimize_code!(expr)

expr = Meta.parse(replace(string(expr), "SymbolicUtils.Code.create_array(typeof(sinput), nothing, Val{1}(), Val{(2,)}()," => "(" ))


func = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(expr))
func(x, sympnet.params)

@time func(x, sympnet.params)
