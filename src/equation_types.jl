# define custom equation type
const EqT = Union{Symbolics.Arr{Num}, AbstractArray{Num}, Num, AbstractArray{<:Symbolics.BasicSymbolic}}