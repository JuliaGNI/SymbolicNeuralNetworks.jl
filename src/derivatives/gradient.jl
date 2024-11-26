struct Gradient{ST, OT, SDT} <: Derivative{ST, OT, SDT} 
    nn::ST
    output::OT
    ∇::SDT
end

derivative(g::Gradient) = g.∇