############################################
#           Abstract Class
############################################

abstract type AbstractSymbolFunction{M,N,S} end

dim_input(::AbstractSymbolFunction{M,N,S}) where {M,N,S} = M
dim_output(::AbstractSymbolFunction{M,N,S}) where {M,N,S} = N
symbol(::AbstractSymbolFunction{M,N,S}) where {M,N,S} = S

function Base.show(io::IO, ::AbstractSymbolFunction{M,N,S}) where {M,N,S}
    print(io, S)
end

############################################
#           Utils for Symbol
############################################

function add_brace_extr(s::Symbol)
    str = string(s) 
    if str[begin] != '(' && str[end] != ')' 
        return Symbol(string("(",s,")"))
    end
    s
end

function remove_brace_extr(s::Symbol)
    str = string(s) 
    if str[begin] == '(' && str[end] == ')'
        return Symbol(str[begin+1:end-1])
    end
    s
end

function add_minus_extr(s::Symbol)
    str = string(s) 
    if str[begin] != '-'
        return Symbol(string("-",str))
    end
    s
end

############################################
#           basic Function
############################################

struct SymbolFunction{M,N,S} <: AbstractSymbolFunction{M,N,S}
    SymbolFunction(s::Symbol, m::Int, n::Int) = new{m,n,s}()
end


############################################
#           IndexSymbolFunction
############################################

struct IndexSymbolFunction{M,S} <: AbstractSymbolFunction{M,1,S}
    f::AbstractSymbolFunction
    index
    function IndexSymbolFunction(f::AbstractSymbolFunction{M,N,S}, index::Symbol) where {M,N,S}
        new{m, Symbol(string(S,"[",index,"]"))}(f, index)
    end
    function IndexSymbolFunction(f::AbstractSymbolFunction{M,N,S}, index::Int) where {M,N,S}
        @assert 1≤index≤N string(f, " doesn't have a ",index,"ᵗʰ component.")
        str_index = N == 1 ? "" : string("[",index,"]")
        new{M, Symbol(string(S, str_index))}(f, index)
    end
end

function Base.getindex(f::AbstractSymbolFunction{M,N,S}, s::Symbol) where {M,N,S}
    IndexSymbolFunction(f,s)
end

function Base.getindex(f::AbstractSymbolFunction{M,N,S}, i::Int) where {M,N,S}
    IndexSymbolFunction(f,i)
end

function Base.getindex(::IndexSymbolFunction{M,S}, ::Symbol) where {M,S}
    string("You cannot index a vector of function more than once.")
end


############################################
#      Abstract OperationSymbolFunction
############################################

abstract type OperationSymbolFunction{M,N,S} <:  AbstractSymbolFunction{M,N,S} end


############################################
#               Addition
############################################

struct AddSymbolFunction{M,N,S} <: OperationSymbolFunction{M,N,S}
    left::AbstractSymbolFunction
    right::AbstractSymbolFunction
    function AddSymbolFunction(f::AbstractSymbolFunction{M,N},g::AbstractSymbolFunction{Q,P}) where {M,N,Q,P}
        @assert N ==  P  string("You cannot add two vectors of different dimension.")
        str_f = remove_brace_extr(symbol(f))
        str_g = remove_brace_extr(symbol(g))
        str_op = typeof(g) <: MinusSymbolFunction ? "" : "+"
        new{M,N,Symbol(string(str_f,str_op,str_g))}(f,g)

    end
end

function Base.:+(f::AbstractSymbolFunction, g::AbstractSymbolFunction)
    AddSymbolFunction(f,g)
end

############################################
#                  Minus
############################################

struct MinusSymbolFunction{M,N,S} <: AbstractSymbolFunction{M,N,S}
    function MinusSymbolFunction(f::AbstractSymbolFunction{M,N,S}) where {M,N,S}
        str_f = typeof(f) <: SymbolFunction ? symbol(f) : add_brace_extr(symbol(f))
        str_f = add_minus_extr(str_f)
        new{M,N,str_f}()
    end
end

function Base.:-(f::AbstractSymbolFunction)
    MinusSymbolFunction(f)
end

function Base.:-(f::AbstractSymbolFunction, g::AbstractSymbolFunction)
    AddSymbolFunction(f, MinusSymbolFunction(g))
end

############################################
#               Multiplication
############################################

struct MultSymbolFunction{M,N,S} <: OperationSymbolFunction{M,N,S}
    left::AbstractSymbolFunction
    right::AbstractSymbolFunction
    function MultSymbolFunction(f::AbstractSymbolFunction{M,N},g::AbstractSymbolFunction{Q,P}) where {M,N,Q,P}
        @assert N == 1 || P == 1 string("You cannot multiply two vectors of dimension higher than two.")
        str_f = typeof(f)<:MultSymbolFunction ? remove_brace_extr(symbol(f)) : f
        str_g = typeof(g)<:MultSymbolFunction ? remove_brace_extr(symbol(g)) : g
        new{M,max(N,P),Symbol(string("(",str_f,*,str_g,")"))}(f,g)
    end
end

function Base.:*(f::AbstractSymbolFunction, g::AbstractSymbolFunction)
    MultSymbolFunction(f,g)
end

############################################
#               Composition
############################################

struct ComposedSymbolFunction{M,N,S} <: OperationSymbolFunction{M,N,S}
    outer::AbstractSymbolFunction
    inner::AbstractSymbolFunction
    function ComposedSymbolFunction(f::AbstractSymbolFunction{M,N},g::AbstractSymbolFunction{Q,P}) where {M,N,Q,P}
        @assert N == Q string("Dimension of output of ",f," must match with the dimension of intput of ",g,".")
        str_f = typeof(f)<:ComposedSymbolFunction ? remove_brace_extr(symbol(f)) : f
        str_g = typeof(g)<:ComposedSymbolFunction ? remove_brace_extr(symbol(g)) : g
        new{M,P,Symbol(string("(",str_g,∘,str_f,")"))}(g,f)
    end
end

function Base.ComposedFunction(g::AbstractSymbolFunction{Q,P}, f::AbstractSymbolFunction{M,N}) where {M,N,Q,P}
    ComposedSymbolFunction(f,g)
end

function Base.ComposedFunction(g::MultSymbolFunction, f::AbstractSymbolFunction)
    MultSymbolFunction(ComposedSymbolFunction(f,g.left),ComposedSymbolFunction(f,g.right))
end

function Base.ComposedFunction(g::AddSymbolFunction, f::AbstractSymbolFunction)
    AddSymbolFunction(ComposedSymbolFunction(f,g.left),ComposedSymbolFunction(f,g.right))
end

############################################
#               Derivative
############################################

abstract type Derivative{D} end

symbol(::Derivative{D}) where D = D

struct DerivativeSymbolFunction{M,N,S} <: AbstractSymbolFunction{M,N,S}
    D::Derivative
    f::AbstractSymbolFunction
    function DerivativeSymbolFunction(d::Derivative{D}, f::AbstractSymbolFunction{M,N,S}) where {M,N,S,D}
        str_f = typeof(f) <: SymbolFunction ? "" : string("(",S,")")
        new{M,N,Symbol(string(D,str_f))}(d,f)
    end
end

struct PartialDerivative{D, I} <: Derivative{D}
    PartialDerivative(i::Int) = new{Symbol(string("∂[",i,"]")),i}()
end

function (pard::PartialDerivative{D,I})(f::AbstractSymbolFunction{M,N,S}) where{D,I,M,N,S}
    @assert 1≤I≤M string(f, " doesn't have a ", I,"ᵗʰ input.")
    DerivativeSymbolFunction(pard, f)
end

function (pard::PartialDerivative{D,I})(f::AbstractSymbolFunction{M,N,S}, j::Int) where{D,I,M,N,S}
    @assert 1≤j≤N string(f, " doesn't have a ",j,"ᵗʰ component.")
    @assert 1≤I≤M string(f, " doesn't have a ",I,"ᵗʰ input.")
    DerivativeSymbolFunction(pard, f[j])
end

function (pard::PartialDerivative{D,I})(f::AbstractSymbolFunction{M,1,S}) where{D,I,M,S}
    @assert 1≤I≤M string(f, " doesn't have a ", I,"ᵗʰ input.")
    pard(f,1)
end

function (pard::PartialDerivative{D,I})(scf::ComposedSymbolFunction{M,N}, j::Int) where{D,I,M,N}
    @assert 1≤j≤N string(scf, " doesn't have a ",j,"ᵗʰ component.")
    @assert 1≤I≤M string(scf, " doesn't have a ",I,"ᵗʰ input.")
    MultSymbolFunction(pard(scf.outer, j)∘scf.inner, pard(scf.inner,j))
end

function (pard::PartialDerivative{D,I})(scf::MultSymbolFunction{M,N}, j::Int) where{D,I,M,N}
    @assert 1≤j≤N string(scf, " doesn't have a ",j,"ᵗʰ component.")
    @assert 1≤I≤M string(scf, " doesn't have a ",I,"ᵗʰ input.")
    AddSymbolFunction(MultSymbolFunction(pard(scf.right, j), scf.left), MultSymbolFunction(scf.right, pard(scf.left, j)))
end

function (pard::PartialDerivative{D,I})(scf::AddSymbolFunction{M,N}, j::Int) where{D,I,M,N}
    @assert 1≤j≤N string(scf, " doesn't have a ",j,"ᵗʰ component.")
    @assert 1≤I≤M string(scf, " doesn't have a ",I,"ᵗʰ input.")
    AddSymbolFunction(pard(scf.right, j), pard(scf.left, j))
end


f = SymbolFunction(:f,1,1)
g = SymbolFunction(:g,1,1)
h = SymbolFunction(:h,1,1)
Dx₁ = PartialDerivative(1)
@time Dx₁(Dx₁(f∘g, 1),1)