abstract type Derivative{ST, SDT} end

derivative(::DT) where {DT <: Derivative} = error("No method of function `derivative` defined for type $(DT).")