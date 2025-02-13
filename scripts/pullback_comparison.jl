using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using GeometricMachineLearning
using AbstractNeuralNetworks: FeedForwardLoss
using GeometricMachineLearning: ZygotePullback
import Random
Random.seed!(123)

c = Chain(Dense(2, 3, tanh), Dense(3, 1, tanh))
nn = SymbolicNeuralNetwork(c)
nn_cpu = NeuralNetwork(c, CPU())
loss = FeedForwardLoss()
spb = SymbolicPullback(nn, loss)
zpb = ZygotePullback(loss)

batch_size = 10000
input = rand(2, batch_size)
output = rand(1, batch_size)
# output sensitivities
_do = 1.

# spb(params(nn_cpu), nn.model, (input, output))[2](_do)
# zpb(params(nn_cpu), nn.model, (input, output))[2](_do)
# @time spb_evaluated = spb(params(nn_cpu), nn.model, (input, output))[2](_do)
# @time zpb_evaluated = zpb(params(nn_cpu), nn.model, (input, output))[2](_do)[1].params
# @assert values(spb_evaluated) .≈ values(zpb_evaluated)

function timenn(pb, params, model, input, output, _do = 1.)
    pb(params, model, (input, output))[2](_do)
    @time pb(params, model, (input, output))[2](_do)
end

timenn(spb, params(nn_cpu), nn.model, input, output)
timenn(zpb, params(nn_cpu), nn.model, input, output)
