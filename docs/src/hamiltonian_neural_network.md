# Hamiltonian Neural Network

Here we build a Hamiltonian neural network as a symbolic neural network.

```@example
using SymbolicNeuralNetworks
using AbstractNeuralNetworks: Dense, initialparameters
import Symbolics

input_dim = 2
d = Dense(input_dim, 1, tanh)

# nn = HamiltonianSymbolicNeuralNetwork(d)

# nn.equations.hvf
```