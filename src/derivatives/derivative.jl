"""
    Derivative
"""
abstract type Derivative{ST, OT, SDT} end

derivative(::DT) where {DT <: Derivative} = error("No method of function `derivative` defined for type $(DT).")

function symbolic_differentials(sparams::Symbolics.Arr)
    collect(Differential.(sparams))
end

function symbolic_differentials(sparams::NamedTuple)
    differential_values = (symbolic_differentials(sparams[key]) for key in keys(sparams))
    NamedTuple{keys(sparams)}(differential_values)
end

function symbolic_differentials(sparams::NeuralNetworkParameters)
    vals = Tuple(symbolic_differentials(sparams[key]) for key in keys(sparams))
    NeuralNetworkParameters{keys(sparams)}(vals)
end

function symbolic_derivative(soutput, Dx::AbstractArray)
    [expand_derivatives(Symbolics.scalarize(dx(soutput))) for dx in Dx]
end

function symbolic_derivative(soutput, dps::NamedTuple)
    gradient_values = (symbolic_gradient(soutput, dps[key]) for key in keys(dps))
    NamedTuple{keys(dps)}(gradient_values)
end

function symbolic_derivative(soutput, dps::NeuralNetworkParameters)
    vals = Tuple(symbolic_gradient(soutput, dp) for dp in values(dps))
    NeuralNetworkParameters{keys(dps)}(vals)
end