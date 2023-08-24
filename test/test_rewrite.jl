using SymbolicNeuralNetworks
using Symbolics
using Test

@variables W1[1:2,1:2] W2[1:2,1:2] b1[1:2] b2[1:2]
sparams = ((W =  W1, b = b1), (W = W2, b = b2))

@variables SX, SY
sargs = [SX, SY]

output = sparams[2].W  * tanh.(sparams[1].W * sargs + sparams[1].b) + sparams[2].b

code_output = build_function(output, sargs..., develop(sparams)...)[2]
rewrite_ouput = eval(rewrite_code(code_output, Tuple(sargs), sparams, "OUTPUT"))

params = ((W = [1 3; 2 2], b = [1, 0]), (W = [1 1; 0 2], b = [1, 0]))
args = [1, 0.2]

@test_nowarn rewrite_ouput(args, params) 


