struct Gradient{ST, SDT} <: Derivative{ST, SDT} 
    nn::ST
    ∇::SDT
end

derivative(g::Gradient) = g.∇