function (::FeedForwardLoss)(model::Union{AbstractNeuralNetworks.Chain,
                                        AbstractNeuralNetworks.AbstractExplicitLayer},
                                        params::NeuralNetworkParameters,
                                        input::EqT,
                                        output::EqT)
        # NOTE: divides by norm(output) — evaluating the built function with a
        # zero-valued output target produces NaN/Inf. Ensure targets are non-zero.
        norm((model(input, params) |> collect) - output) / norm(output)
end
