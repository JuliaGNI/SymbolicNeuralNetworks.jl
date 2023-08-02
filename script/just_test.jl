using Symbolics


using GeometricMachineLearning
include("symbolic_params.jl")

#=
hnn = NeuralNetwork(HamiltonianNeuralNetwork(2; nhidden = 0), Float32)

dimin = dim(hnn.architecture)



# creates variables for the input
@variables sinput[1:dimin]

# creates variables for the parameters
sparams = symbolicParams(hnn)

hmm(x, params) = hnn(x, params)[1]

est = Symbolics.scalarize(hmm(sinput, sparams))

est2 = hmm(sinput, sparams)

field_ =  Symbolics.gradient(est, sinput)

fun_est = build_function(est, sinput, develop(sparams)...)
fun_est2 = build_function(est2, sinput, develop(sparams)...)
fun_field = build_function(field_, sinput, develop(sparams)...)[1]

#write("src/field.jl", get_string(fun_field))

ffield = eval(fun_field)
params2 =  develop(hnn.params)
@time ffield([0.2,1.2], params2...)
fest = eval(fun_est)
fest2 =  eval(fun_est2)
@time fest([0.2,1.2], params2...)
@time fest2([0.2,1.2], params2...)
@time hmm([0.2,1.2], hnn.params)
import GeometricMachineLearning: vectorfield
@time vectorfield(hnn, [0.2,1.2])

include("field.jl")
@time fieldd([0.2,1.2], params2...)

=#
function f()
    @variables x, y
    x^2+y
end

z= f()
@variables x 

D=Differential(x)
expand_derivatives(D(z))
