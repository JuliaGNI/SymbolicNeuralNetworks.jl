using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")
include("symbolic_params.jl")

#####

abstract type AbstractNeuralNetwork end

abstract type AbstractSymbolicNeuralNetwork{AT, ET} <: AbstractNeuralNetwork end


