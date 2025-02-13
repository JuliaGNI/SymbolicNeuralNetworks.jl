"""
    substitute_gradient(eq, s∇nn, s∇output)

Substitute the symbolic expression `s∇nn` in `eq` with the symbolic expression `s∇output`.

# Implementation 

See the comment in [`evaluate_equation`](@ref).
"""
function substitute_gradient(eq, s∇nn, s∇output)
    @assert axes(s∇nn) == axes(s∇output)
    substitute.(eq, Ref(Dict([s∇nn[i] => s∇output[i] for i in axes(s∇nn, 1)])))
end

function substitute_gradient(eq, ::Nothing, ::Nothing)
    eq 
end

"""
    evaluate_equation(eq, soutput)

Replace `snn` in `eq` with `soutput` (input), scalarize and expand derivatives.

# Implementation

Here we use `Symbolics.substitute` with broadcasting to be able to handle `eq`s that are arrays.
For that reason we use [`Ref` before `Dict`](https://discourse.julialang.org/t/symbolics-and-substitution-using-broadcasting/68705).
This is also the case for the functions [`substitute_gradient`](@ref).
"""
function evaluate_equation(eq::EqT, snn::EqT, s∇nn::EqT, soutput::EqT, s∇output::EqT)
    @assert axes(snn) == axes(soutput)
    eq_output_substituted = substitute.(eq, Ref(Dict([snn[i] => soutput[i] for i in axes(snn, 1)])))
    substitute_gradient(eq_output_substituted, s∇nn, s∇output)
end

"""
    evaluate_equations(eqs, soutput)

Apply [`evaluate_equation`](@ref) to a `NamedTuple` and append `(soutput = soutput, s∇output = s∇output)`.
"""
function evaluate_equations(eqs::NamedTuple, snn::EqT, s∇nn::EqT, soutput::EqT, s∇output::EqT; simplify = true)
    
    # closure
    _evaluate_equation(eq) = evaluate_equation(eq, snn, s∇nn, soutput, s∇output)
    evaluated_equations = Tuple(_evaluate_equation(eq) for eq in eqs)

    soutput_eq = (soutput = simplify == true ? Symbolics.simplify(soutput) : soutput,)
    s∇output_eq = isnothing(s∇nn) ? NamedTuple() : (s∇output = simplify == true ? Symbolics.simplify(s∇output) : s∇output,)
    merge(soutput_eq, s∇output_eq, NamedTuple{keys(eqs)}(evaluated_equations))
end

"""
    evaluate_equations(eqs, nn)

Expand the output and gradient in `eqs` with the weights in `nn`.

- `eqs` here has to be a `NamedTuple` that contains keys 
- `:x`: gives the inputs to the neural network and 
- `:nn`: symbolic expression of the neural network.

# Implementation

Internally this 
1. computes the gradient and
2. calls [`evaluate_equations(::NamedTuple, ::EqT, ::EqT, ::EqT, EqT)`](@ref).

"""
function evaluate_equations(eqs::NamedTuple, nn::AbstractSymbolicNeuralNetwork; kwargs...)
    @assert [:x, :nn] ⊆ keys(eqs)
    
    sinput = eqs.x

    snn = eqs.nn

    s∇nn = haskey(eqs, :∇nn) ? eqs.∇nn : nothing

    remaining_eqs = NamedTuple([p for p in pairs(eqs) if p[1] ∉ [:x, :nn, :∇nn]])

    # Evaluation of the symbolic output
    soutput = _scalarize(nn.model(sinput, params(nn)))

    # make differential 
    Dx = symbolic_differentials(sinput)

    # Evaluation of gradient
    s∇output = isnothing(s∇nn) ? nothing : [expand_derivatives(Symbolics.scalarize(dx(soutput))) for dx in Dx]

    evaluate_equations(remaining_eqs, snn, s∇nn, soutput, s∇output; kwargs...)
end