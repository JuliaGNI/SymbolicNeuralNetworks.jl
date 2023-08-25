using GeometricMachineLearning
using SymbolicNeuralNetworks
using Symbolics
using KernelAbstractions
using Test

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
@test functions(ssympnet).eval(x, sympnet.params) == sympnet(x)


@time functions(ssympnet).eval(x, sympnet.params)
@time sympnet(x)
#=
@kernel function assign_first_half!(q::AbstractVector, x::AbstractVector)
    i = @index(Global)
    q[i] = x[i]
end

@kernel function assign_second_half!(p::AbstractVector, x::AbstractVector, N::Integer)
    i = @index(Global)
    p[i] = x[i+N]
end

@kernel function assign_first_half!(q::AbstractMatrix, x::AbstractMatrix)
    i,j = @index(Global, NTuple)
    q[i,j] = x[i,j]
end

@kernel function assign_second_half!(p::AbstractMatrix, x::AbstractMatrix, N::Integer)
    i,j = @index(Global, NTuple)
    p[i,j] = x[i+N,j]
end

@kernel function assign_first_half!(q::AbstractArray{T, 3}, x::AbstractArray{T, 3}) where T 
    i,j,k = @index(Global, NTuple)
    q[i,j,k] = x[i,j,k]
end

@kernel function assign_second_half!(p::AbstractArray{T, 3}, x::AbstractArray{T, 3}, N::Integer) where T 
    i,j,k = @index(Global, NTuple)
    p[i,j,k] = x[i+N,j,k]
end

function assign_q_and_p(x::AbstractVector, N)
    backend = try KernelAbstractions.get_backend(x)
    catch 
        CPU() end
    q = KernelAbstractions.allocate(backend, eltype(x), N)
    p = KernelAbstractions.allocate(backend, eltype(x), N)
    q_kernel! = assign_first_half!(backend)
    p_kernel! = assign_second_half!(backend)
    q_kernel!(q, x, ndrange=size(q))
    p_kernel!(p, x, N, ndrange=size(p))
    (q, p)
end

assign_q_and_p(x, 1)
=#