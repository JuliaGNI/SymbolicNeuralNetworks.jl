"""
    Derivative

Supertype of the symbolic derivatives this package computes: [`Jacobian`](@ref) (with respect to the
input of a network) and [`Gradient`](@ref) (with respect to its parameters). Use
[`derivative`](@ref) to get the symbolic expression out of one.
"""
abstract type Derivative{OT, SDT, ST <: AbstractSymbolicNeuralNetwork} end

"""
    derivative(d)

The symbolic derivative stored in `d`.
"""
derivative(::DT) where {DT <: Derivative} = error("No method of function `derivative` defined for type $(DT).")

"""
    symbolic_differentials(svariables)

The differential operators belonging to a set of symbolic variables, with the same shape and
nesting as the variables themselves. See [`symbolic_derivative`](@ref).
"""
symbolic_differentials(svariables::AbstractArray) = Differential.(svariables)
symbolic_differentials(svariables::Symbolics.Arr) = symbolic_differentials(collect(svariables))

symbolic_differentials(svariables::NamedTuple) =
    NamedTuple{keys(svariables)}(map(symbolic_differentials, values(svariables)))
symbolic_differentials(svariables::NeuralNetworkParameters) =
    NeuralNetworkParameters{keys(svariables)}(map(symbolic_differentials, values(svariables)))

"""
    symbolic_derivative(f, differentials)

Differentiate the scalar expression `f` with the differential operators in `differentials`, keeping
their shape and nesting. Together with [`symbolic_differentials`](@ref) this is what turns "the
parameters of a network" into "the derivative of `f` with respect to each of them".
"""
symbolic_derivative(f, differentials::AbstractArray) = [expand_derivatives(D(f)) for D in differentials]

symbolic_derivative(f, differentials::NamedTuple) =
    NamedTuple{keys(differentials)}(map(D -> symbolic_derivative(f, D), values(differentials)))
symbolic_derivative(f, differentials::NeuralNetworkParameters) =
    NeuralNetworkParameters{keys(differentials)}(map(D -> symbolic_derivative(f, D), values(differentials)))
