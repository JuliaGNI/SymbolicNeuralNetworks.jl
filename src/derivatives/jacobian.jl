struct Jacobian{ST, SDT} <: Derivative{ST, SDT} 
    nn::ST
    □::SDT
end

derivative(j::Jacobian) = j.□