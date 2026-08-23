# `AbstractNeuralNetworks.FeedForwardLoss` is only defined for numeric inputs. This method extends it
# to symbolic ones, which is what lets a `NetworkLoss` be differentiated symbolically by
# `SymbolicPullback`. It is deliberately a method on a type owned by `AbstractNeuralNetworks`; the
# alternative would be to duplicate every loss of that package here.
"""
    (::FeedForwardLoss)(model, params, input, output)

The `FeedForwardLoss` of `AbstractNeuralNetworks`, evaluated on symbolic arguments.

!!! warning "Zero targets give `NaN`"
    The loss is normalised by `norm(output)`, so a target that is identically zero makes the
    generated function return `NaN`/`Inf`.
"""
function (::FeedForwardLoss)(model::Union{AbstractNeuralNetworks.Chain, AbstractNeuralNetworks.AbstractExplicitLayer},
                             params::NetworkParameters,
                             input::SymbolicExpression,
                             output::SymbolicExpression)
    norm(scalar_expressions(model(input, params)) - scalar_expressions(output)) / norm(scalar_expressions(output))
end
