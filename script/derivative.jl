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
#           Abstract Class
############################################

abstract type AbstractSymbolicFunction{M,N,Q} end

const AbstractScalarSymbolicFunction{M}   = AbstractSymbolicFunction{M,1,1} where M
const AbstractVectorSymbolicFunction{M,N} = AbstractSymbolicFunction{M,N,1} where {M,N}

abstract type OperationSymbolicFunction{M,N,Q} <: AbstractSymbolicFunction{M,N,Q} end

dimIn(::AbstractSymbolicFunction{M,N,Q}) where {M,N,Q} = M
laxes(::AbstractSymbolicFunction{M,N,Q}) where {M,N,Q} = (N,Q)
symbol(f::AbstractSymbolicFunction) = f.s

function Base.show(io::IO, f::AbstractSymbolicFunction)
    print(io, remove_brace_extr(symbol(f)))
end

############################################
#           Basic Function
############################################

struct MatrixSymbolicFunction{M,N,Q} <: AbstractSymbolicFunction{M,N,Q}
    s::Symbol
    MatrixSymbolicFunction(s::Symbol, m::Int, n::Int, p::Int) = new{m,n,p}(s)
end

const ScalarSymbolicFunction{M}   = MatrixSymbolicFunction{M,1,1} where M
const VectorSymbolicFunction{M,N} = MatrixSymbolicFunction{M,N,1} where {M,N}

ScalarSF(s::Symbol, m::Int) = MatrixSymbolicFunction(s,m,1,1)
VectorSF(s::Symbol, m::Int, n::Int) = MatrixSymbolicFunction(s,m,n,1)
MatrixSF(s::Symbol, m::Int, n::Int, p::Int) = MatrixSymbolicFunction(s,m,n,p)

############################################
#           IndexSymbolFunction
############################################

#=

const typeIndex = Union{Int,Symbol}

projectTo(n::Int,q::Int) = n == 1 ? (1,1) : (1,q)

struct IndexSymbolicFunction{M,N,Q} <: AbstractSymbolicFunction{M,N,Q}
    s::Symbol
    f::AbstractSymbolFunction
    index
    function IndexSymbolicFunction(f::AbstractSymbolicFunction{M,N,Q}, index₁::typeIndex, index₂::typeIndex) where {M,N,Q}
        if typeof(index₁) != Symbol
            @assert 1≤index₁≤N
        end
        if typeof(index₂) != Symbol
            @assert 1≤index₂≤Q
        end

        if 
        str_index1 = 
        new{M,1,1}(Symbol(string(symbol(f),"[",index₁,",",index₂,"]")), f, index)
    end
    function IndexSymbolicFunction(f::AbstractSymbolicFunction{M,N,Q}, index::typeIndex) where {M,N,Q}
        if typeof(index)<:Int
            @assert 1≤index≤N string(f, " doesn't have a ",i,"ᵗʰ component.")
        end
        str_index = N==1 ? "" : string("[",index,"]")
        new{M,1,Q}(Symbol(string(symbol(f), str_index)), f, index)
    end
end

function Base.getindex(f::AbstractSymbolicFunction{M,N,Q}, s::typeIndex...) where {M,N,Q}
    
    IndexSymbolFunction(f,s)
end

=#

############################################
#               Addition
############################################

struct AddSymbolicFunction{M,N,Q} <: OperationSymbolicFunction{M,N,Q}
    s::Symbol
    left::AbstractSymbolicFunction
    right::AbstractSymbolicFunction
    function AddSymbolicFunction(f::AbstractSymbolicFunction{M,N,Q}, g::AbstractSymbolicFunction{P,R,S}) where {M,N,Q,P,R,S}
        @assert M == P && N ==  R && Q == S
        str_f = remove_brace_extr(symbol(f))
        str_g = remove_brace_extr(symbol(g))
        str_op = typeof(g) <: MinusSymbolicFunction ? "" : "+"
        new{M,N,Q}(Symbol(string(str_f,str_op,str_g)), f,g)
    end
end

function Base.:+(f::AbstractSymbolicFunction, g::AbstractSymbolicFunction)
    AddSymbolicFunction(f,g)
end

############################################
#                  Minus
############################################

struct MinusSymbolicFunction{M,N,Q} <: AbstractSymbolicFunction{M,N,Q}
    s::Symbol
    function MinusSymbolicFunction(f::AbstractSymbolicFunction{M,N,Q}) where {M,N,Q}
        str_f = typeof(f) <: MatrixSymbolicFunction ? symbol(f) : add_brace_extr(symbol(f))
        str_f = add_minus_extr(str_f)
        new{M,N,Q}(str_f)
    end
end

function Base.:-(f::AbstractSymbolicFunction)
    MinusSymbolicFunction(f)
end

function Base.:-(f::AbstractSymbolicFunction, g::AbstractSymbolicFunction)
    AddSymbolicFunction(f, MinusSymbolicFunction(g))
end

############################################
#               Multiplication
############################################

struct MultSymbolicFunction{M,N,Q} <: OperationSymbolicFunction{M,N,Q}
    s::Symbol
    left::AbstractSymbolicFunction
    right::AbstractSymbolicFunction
    function MultSymbolicFunction(f::AbstractSymbolicFunction{M,N,Q},g::AbstractSymbolicFunction{P,R,S}) where {M,N,Q,P,R,S}
        @assert M == P || R == Q
        str_f = typeof(f)<:MultSymbolicFunction ? remove_brace_extr(symbol(f)) : symbol(f)
        str_g = typeof(g)<:MultSymbolicFunction ? remove_brace_extr(symbol(g)) : symbol(g)
        new{M,N,S}(Symbol(string("(",str_f,*,str_g,")")),f,g)
    end
end

function Base.:*(f::AbstractSymbolicFunction, g::AbstractSymbolicFunction)
    MultSymbolicFunction(f,g)
end

############################################
#               Composition
############################################

struct ComposedSymbolicFunction{M,N,Q} <: OperationSymbolicFunction{M,N,Q}
    s::Symbol
    outer::AbstractSymbolicFunction
    inner::AbstractSymbolicFunction
    function ComposedSymbolicFunction(f::AbstractSymbolicFunction{M,N,Q},g::AbstractSymbolicFunction{P,R,S}) where {M,N,Q,P,R,S}
        @assert Q == 1 string("Composition is only supported with inner vector functions.")
        @assert N == P string("Dimension of output of ",f," must match with the dimension of intput of ",g,".")
        str_f = typeof(f)<:ComposedSymbolicFunction ? remove_brace_extr(symbol(f)) : symbol(f)
        str_g = typeof(g)<:ComposedSymbolicFunction ? remove_brace_extr(symbol(g)) : symbol(g)
        new{M,R,S}(Symbol(string("(",str_g,∘,str_f,")")),g,f)
    end
end

function Base.ComposedFunction(g::AbstractSymbolicFunction{Q,P}, f::AbstractSymbolicFunction{M,N}) where {M,N,Q,P}
    ComposedSymbolicFunction(f,g)
end

function Base.ComposedFunction(g::MultSymbolicFunction, f::AbstractSymbolicFunction)
    MultSymbolFunction(ComposedSymbolicFunction(f,g.left),ComposedSymbolicFunction(f,g.right))
end

function Base.ComposedFunction(g::AddSymbolicFunction, f::AbstractSymbolicFunction)
    AddSymbolicFunction(ComposedSymbolicFunction(f,g.left),ComposedSymbolicFunction(f,g.right))
end

############################################
#               Derivative
############################################

struct Differential{P,M,N,Q} 
    Differential(p::Int,m::Int,n::Int,q::Int)=new{p,m,n,q}() 
end

symbol(::Differential) = :d 

const Derivative    = Differential{0,1,1,1}
const Gradient{M}   = Differential{1,M,1,1} where M
const Jacobian{M,N} = Differential{1,M,N,1} where {M,N}
const Hessian{M}    = Differential{2,M,1,1} where M

Derivative()            = Differential(0,1,1,1)
Gradient(m::Int)        = Differential(1,m,1,1)
Jacobian(m::Int,n::Int) = Differential(1,m,n,1)
Hessian(m::Int)         = Differential(2,m,1,1)

Differential(p::Int, ::AbstractSymbolicFunction{M,N,Q}) where {M,N,Q} = Differential(p,M,N,Q)

symbol(::Derivative) = :DeriV_
symbol(::Gradient)   = :∇
symbol(::Jacobian)   = :Jac_
symbol(::Hessian)    = :Hess_ 

function push(m::Int,n::Int,q::Int, it::Int = 1)
    if it == 0
        return (m,n,q)
    elseif q == 1 && n == 1
        return push(m,m,1,it-1)
    elseif q == 1
        return push(m,m,n,it-1)
    end
    @error string("Push of (",m,",",n,",",q,") is not yet supported.")
end

struct DifferentialSymbolicFunction{TD,M,N,Q} <: AbstractSymbolicFunction{M,N,Q}
    s::Symbol
    D::TD
    f::AbstractSymbolicFunction
    function DifferentialSymbolicFunction(d::Differential{P}, f::AbstractSymbolicFunction{M,N,Q},dim::Tuple) where{P,M,N,Q}
        str_f = typeof(f) <: MatrixSymbolicFunction ? symbol(f) : string("(",symbol(f),")")
        new{typeof(d), dim...}(Symbol(string(symbol(d),str_f)),d,f)
    end
end

(d::Differential{P})(f::AbstractSymbolicFunction{M,N,Q}) where {P,M,N,Q} = DifferentialSymbolicFunction(d,f,push(M,N,Q,P))
(d::Differential{P1})(f::DifferentialSymbolicFunction{<:Differential,M,N,Q}) where {P1,M,N,Q} = DifferentialSymbolicFunction(_concat(d,f.D),f.f,push(M,N,Q,P1))

_concat(::Differential{P1}, ::Differential{P2,M,N,Q}) where {P1,P2,M,N,Q} = Differential(P1+P2,M,N,Q)

Derivative(::AbstractSymbolicFunction) = Differential(0,f)(f)
Gradient(::AbstractSymbolicFunction)   = Differential(1,f)(f)
Jacobian(::AbstractSymbolicFunction)   = Differential(1,f)(f)
Hessian(::AbstractSymbolicFunction)    = Differential(2,f)(f)

############################################
#               PartialDerivative
############################################

struct PartialDerivative <: Derivative
    i::Int
    PartialDerivative(i::Int) = new(i)
end

symbol(pard::PartialDerivative) = Symbol(string("∂[",pard.i,"]"))

function (pard::PartialDerivative)(f::AbstractSymbolicFunction{M,N,Q}) where{M,N,Q}
    @assert 1≤pard.i≤M string(f, " doesn't have a ", i,"ᵗʰ input.")
    DerivativeSymbolicFunction(pard, f,(M,N,Q))
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

