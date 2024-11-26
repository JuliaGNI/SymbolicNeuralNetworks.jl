abstract type Derivative end

derivative(::DT) where {DT <: Derivative} = error("No method of function `derivative` defined for type $(DT).")