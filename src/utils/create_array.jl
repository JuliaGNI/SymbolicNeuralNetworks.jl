function Symbolics.SymbolicUtils.Code.create_array(::Type{<:Base.ReshapedArray{T, N, P}}, S, nd::Val, d::Val, elems...) where {T, N, P}
    Symbolics.SymbolicUtils.Code.create_array(P, S, nd, d, elems...)
end